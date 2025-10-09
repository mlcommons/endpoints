#!/usr/bin/env python3
"""
Custom h2load stress test runner with flexible configuration.
Allows easy parameterization and result analysis.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class H2LoadTestConfig:
    """Configuration for a single h2load test."""

    name: str
    clients: int
    threads: int
    requests: int
    max_concurrent: int
    duration: int = 0  # 0 means use requests instead
    protocol: str = "http/1.1"  # "http/1.1" or "http/2"

    def validate(self):
        """Validate test configuration."""
        if self.clients <= 0:
            raise ValueError("clients must be > 0")
        if self.threads <= 0:
            raise ValueError("threads must be > 0")
        if self.threads > self.clients:
            raise ValueError("threads cannot exceed clients")
        if self.duration == 0 and self.requests <= 0:
            raise ValueError("either duration or requests must be specified")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")


class H2LoadRunner:
    """Runner for h2load stress tests."""

    def __init__(
        self,
        server_url: str,
        endpoint: str = "/v1/chat/completions",
        output_dir: str = "./h2load_results",
        payload: dict[str, Any] | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.endpoint = endpoint
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default payload
        self.payload = payload or {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "What is the capital of France? " * 10,  # Make it bigger
                },
            ],
            "stream": False,
        }

        # Save payload to file
        self.payload_file = self.output_dir / "request_payload.json"
        with open(self.payload_file, "w") as f:
            json.dump(self.payload, f, indent=2)

        self.results: list[dict[str, Any]] = []

    def check_h2load_installed(self) -> bool:
        """Check if h2load is installed."""
        try:
            subprocess.run(
                ["h2load", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def run_test(self, config: H2LoadTestConfig) -> dict[str, Any]:
        """Run a single h2load test."""
        config.validate()

        print(f"\n{'='*70}")
        print(f"Running Test: {config.name}")
        print(f"{'='*70}")
        print(f"Clients:              {config.clients}")
        print(f"Threads:              {config.threads}")
        print(
            f"Requests:             {config.requests if config.duration == 0 else 'Duration-based'}"
        )
        print(f"Max Concurrent:       {config.max_concurrent}")
        print(
            f"Duration:             {config.duration}s" if config.duration > 0 else ""
        )
        print(f"Protocol:             {config.protocol}")
        print()

        # Build h2load command
        cmd = [
            "h2load",
            "-c",
            str(config.clients),
            "-t",
            str(config.threads),
            "-m",
            str(config.max_concurrent),
            "-d",
            str(self.payload_file),
            "-H",
            "Content-Type: application/json",
        ]

        # Add duration or request count
        if config.duration > 0:
            cmd.extend(["-D", str(config.duration)])
        else:
            cmd.extend(["-n", str(config.requests)])

        # Add protocol flag
        if config.protocol == "http/1.1":
            cmd.append("--h1")

        # Add URL
        full_url = f"{self.server_url}{self.endpoint}"
        cmd.append(full_url)

        # Output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{config.name}_{timestamp}.txt"

        print(f"Command: {' '.join(cmd)}")
        print(f"Output: {output_file}")
        print()

        # Run the test
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(config.duration + 60, 300) if config.duration > 0 else 600,
            )

            elapsed_time = time.time() - start_time

            # Save output
            with open(output_file, "w") as f:
                f.write("Test Configuration:\n")
                f.write("==================\n")
                for key, value in asdict(config).items():
                    f.write(f"{key}: {value}\n")
                f.write(f"\nCommand: {' '.join(cmd)}\n")
                f.write("\nOutput:\n")
                f.write("=======\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\nErrors:\n")
                    f.write("=======\n")
                    f.write(result.stderr)

            # Parse results
            test_result = self._parse_h2load_output(result.stdout, config, elapsed_time)
            test_result["output_file"] = str(output_file)
            test_result["success"] = result.returncode == 0

            self.results.append(test_result)

            # Print summary
            print(f"✓ Test completed in {elapsed_time:.2f}s")
            print(f"  Requests/sec: {test_result.get('req_per_sec', 'N/A')}")
            print(f"  Success rate: {test_result.get('success_rate', 'N/A')}%")

            return test_result

        except subprocess.TimeoutExpired:
            print("✗ Test timed out")
            return {
                "name": config.name,
                "success": False,
                "error": "timeout",
                "config": asdict(config),
            }
        except Exception as e:
            print(f"✗ Test failed: {e}")
            return {
                "name": config.name,
                "success": False,
                "error": str(e),
                "config": asdict(config),
            }

    def _parse_h2load_output(
        self,
        output: str,
        config: H2LoadTestConfig,
        elapsed_time: float,
    ) -> dict[str, Any]:
        """Parse h2load output to extract metrics."""
        result = {
            "name": config.name,
            "config": asdict(config),
            "elapsed_time": elapsed_time,
        }

        # Extract key metrics using simple parsing
        for line in output.split("\n"):
            line = line.strip()

            # Requests per second
            if "req/s" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "req/s" in part.lower() and i > 0:
                            result["req_per_sec"] = float(parts[i - 1])
                            break
                except (ValueError, IndexError):
                    pass

            # Time to completion
            if "finished in" in line.lower():
                try:
                    # Example: "finished in 5.23s, 1234.56 req/s"
                    parts = line.split()
                    for _, part in enumerate(parts):
                        if part.endswith("s,") or part.endswith("s"):
                            time_str = part.rstrip(",").rstrip("s")
                            result["time_finished"] = float(time_str)
                            break
                except (ValueError, IndexError):
                    pass

            # Status codes
            if "status" in line.lower() and "2xx" in line:
                try:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        result["status_2xx"] = int(parts[1].strip())
                except (ValueError, IndexError):
                    pass

            # Traffic
            if "traffic:" in line.lower():
                try:
                    # Extract total traffic
                    parts = line.split(":")
                    if len(parts) >= 2:
                        result["traffic"] = parts[1].strip()
                except Exception:
                    pass

        # Calculate success rate
        if "status_2xx" in result and config.requests > 0:
            result["success_rate"] = (result["status_2xx"] / config.requests) * 100

        return result

    def run_test_suite(self, configs: list[H2LoadTestConfig], warmup: bool = True):
        """Run a suite of tests."""
        if not self.check_h2load_installed():
            print("ERROR: h2load is not installed!")
            print("\nInstallation instructions:")
            print("  Ubuntu/Debian: sudo apt-get install nghttp2-client")
            print("  CentOS/RHEL:   sudo yum install nghttp2")
            print("  macOS:         brew install nghttp2")
            sys.exit(1)

        print(f"\n{'#'*70}")
        print("h2load Stress Test Suite")
        print(f"{'#'*70}")
        print(f"Server URL:       {self.server_url}{self.endpoint}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Number of Tests:  {len(configs)}")
        print(f"Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}\n")

        # Warmup
        if warmup:
            print("Running warmup test...")
            warmup_config = H2LoadTestConfig(
                name="warmup",
                clients=10,
                threads=2,
                requests=100,
                max_concurrent=10,
                protocol="http/1.1",
            )
            self.run_test(warmup_config)
            time.sleep(2)

        # Run tests
        for i, config in enumerate(configs, 1):
            print(f"\n[Test {i}/{len(configs)}]")
            self.run_test(config)
            time.sleep(2)  # Brief pause between tests

        # Generate summary
        self._generate_summary()

    def _generate_summary(self):
        """Generate summary report of all tests."""
        summary_file = (
            self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        summary = {
            "test_suite": {
                "server_url": f"{self.server_url}{self.endpoint}",
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "successful_tests": sum(
                    1 for r in self.results if r.get("success", False)
                ),
            },
            "results": self.results,
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print("Test Suite Summary")
        print(f"{'='*70}")
        print(f"Total Tests:      {summary['test_suite']['total_tests']}")
        print(f"Successful:       {summary['test_suite']['successful_tests']}")
        print(
            f"Failed:           {summary['test_suite']['total_tests'] - summary['test_suite']['successful_tests']}"
        )
        print(f"\nSummary saved to: {summary_file}")

        # Print performance comparison
        print(f"\n{'='*70}")
        print("Performance Comparison")
        print(f"{'='*70}")
        print(f"{'Test Name':<30} {'Clients':<10} {'RPS':<15} {'Success %':<12}")
        print(f"{'-'*70}")

        for result in self.results:
            if result.get("success"):
                name = result["name"][:28]
                clients = result["config"]["clients"]
                rps = result.get("req_per_sec", "N/A")
                rps_str = f"{rps:.2f}" if isinstance(rps, int | float) else str(rps)
                success_rate = result.get("success_rate", "N/A")
                success_str = (
                    f"{success_rate:.2f}"
                    if isinstance(success_rate, int | float)
                    else str(success_rate)
                )
                print(f"{name:<30} {clients:<10} {rps_str:<15} {success_str:<12}")

        print(f"{'='*70}\n")


def create_standard_test_suite() -> list[H2LoadTestConfig]:
    """Create a standard test suite with progressive load."""
    return [
        H2LoadTestConfig(
            name="01_baseline_low",
            clients=10,
            threads=2,
            requests=1000,
            max_concurrent=10,
        ),
        H2LoadTestConfig(
            name="02_baseline_medium",
            clients=50,
            threads=4,
            requests=5000,
            max_concurrent=50,
        ),
        H2LoadTestConfig(
            name="03_high_concurrency",
            clients=100,
            threads=8,
            requests=10_000,
            max_concurrent=100,
        ),
        H2LoadTestConfig(
            name="04_very_high_concurrency",
            clients=500,
            threads=16,
            requests=50_000,
            max_concurrent=500,
        ),
        H2LoadTestConfig(
            name="05_extreme_concurrency",
            clients=1000,
            threads=32,
            requests=100_000,
            max_concurrent=1000,
        ),
        H2LoadTestConfig(
            name="06_sustained_30s",
            clients=500,
            threads=16,
            requests=0,
            max_concurrent=500,
            duration=30,
        ),
        H2LoadTestConfig(
            name="07_sustained_60s",
            clients=1000,
            threads=32,
            requests=0,
            max_concurrent=1000,
            duration=60,
        ),
    ]


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="h2load Stress Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard test suite
  python h2load_custom_test.py --url http://localhost:12345

  # Run custom single test
  python h2load_custom_test.py --url http://localhost:12345 \\
      --clients 1000 --threads 32 --requests 100000

  # Run with specific endpoint
  python h2load_custom_test.py --url http://localhost:12345 \\
      --endpoint /v1/chat/completions --clients 500

  # Run with custom output directory
  python h2load_custom_test.py --url http://localhost:12345 \\
      --output-dir /tmp/h2load_results
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Server URL (e.g., http://localhost:12345)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="API endpoint (default: /v1/chat/completions)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./h2load_results",
        help="Output directory for results (default: ./h2load_results)",
    )
    parser.add_argument(
        "--standard-suite",
        action="store_true",
        help="Run standard test suite",
    )

    # Custom test parameters
    parser.add_argument(
        "--clients",
        type=int,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads to use",
    )
    parser.add_argument(
        "--requests",
        type=int,
        help="Total number of requests",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Maximum concurrent streams",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Test duration in seconds (0 = use request count)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup test",
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    runner = H2LoadRunner(
        server_url=args.url,
        endpoint=args.endpoint,
        output_dir=args.output_dir,
    )

    # Determine test suite
    if args.standard_suite:
        configs = create_standard_test_suite()
    elif args.clients and args.threads and (args.requests or args.duration):
        # Custom single test
        max_concurrent = args.max_concurrent or args.clients
        requests = args.requests or 0
        configs = [
            H2LoadTestConfig(
                name="custom_test",
                clients=args.clients,
                threads=args.threads,
                requests=requests,
                max_concurrent=max_concurrent,
                duration=args.duration,
            )
        ]
    else:
        # Default to standard suite
        print("No test configuration specified, running standard test suite...")
        configs = create_standard_test_suite()

    # Run tests
    runner.run_test_suite(configs, warmup=not args.no_warmup)


if __name__ == "__main__":
    main()
