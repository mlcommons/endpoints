# Benchmark Hot-Path

This document describes the performance-critical data flow during benchmark execution. The architecture uses three concurrent execution contexts that work together to achieve precise timing control while maximizing throughput.

---

## 1. High-Level Overview

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                               BENCHMARK EXECUTION - DATA FLOW                                             ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    SESSION THREAD            EVENT LOOP THREAD              WORKER PROCESS (N instances)
    (BenchmarkSession)        (uvloop)                       (Separate OS Process)
   ────────────────          ────────────────               ─────────────────────────────

   ┌─────────────┐           ┌─────────────────┐            │              ┌───────────────────────────┐         │           ┌──────────────────┐
   │   LOOP 1    │           │     LOOP 2      │            │            ┌─┤         LOOP 3            │         │           │                  │
   │  Metronome  │           │   Dispatcher    │            │          ┌─┤ │         Engine            │TCP/HTTP │           │  ENDPOINT / SUT  │
   │ (LoadGen)   │──────────▶│ (SampleIssuer)  │────────────┼─────────▶│ │ │    (HTTP Executor)        │─────────┼──────────▶│ (vLLM / TGI etc) │
   └──────┬──────┘  issue()  └────────┬────────┘  IPC(reqs) │          │ └─┴───────────────────────────┘         │           └──────────────────┘
          │                           │                     │          └────────────────────────────┘            │
     (t0: Start)                      │◀────────────────────┼────────────────────────────┘                       │
                                      │           IPC(resps)│                                                    │
                              ┌───────▼────────┐            │                                                    │
                              │ Sample Handler │            │                                                    │
                              │   (Metrics)    │            │                                                    │
                              └───────┬────────┘            │                                                    │
                                  (t1: Stop)                │                                                    │
```

---

## 2. Complete Data Flow (Single Request)

```
SESSION THREAD (MAIN-PROC)
│
│  1. Scheduler yields (sample, delay)
│  2. Busy-wait until target time
│  3. Record Start Time (t0)
│  4. Issue query (handoff to Loop 2) ─────┐
│  5. Continue to next sample              │
│                                          │
                                           │
EVENT LOOP THREAD (MAIN-PROC)              │
│                                          │
│  6. Dispatch to Worker X (Round-robin) ◄─┘
│  7. Push to ZMQ Queue ──────────────────────────┐
│                                                 │
│  11. Receive response from ZMQ ◄────────────────┼────────────────────────┐
│  12. Trigger Event Handlers                     │                        │
│      (Record TTFT / t1 Stop)                    │                        │
│                                                 │                        │
                                                  │                        │
WORKER PROCESS X (SCALE-OUT)                      │                        │           ENDPOINT
│                                                 │                        │              │
│  8. Pull query from IPC ◄───────────────────────┘                        │              │
│  9. HTTP POST ───────────────────────────────────────────────────────────┼─────────────►│
│                                                                          │              │
│  10. Push response to IPC ───────────────────────────────────────────────┘              │
│
```

---
