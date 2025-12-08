# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Evaluator class with fake data."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from inference_endpoint.accuracy.evaluators.base import ABCDExtractor, Evaluator
from inference_endpoint.accuracy.evaluators.scorers import PassAt1Scorer, RougeScorer


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, ground_truths):
        """Initialize with ground truths."""
        self.ground_truths = ground_truths

    def get_ground_truth(self, index, ds):
        """Return ground truth for given index."""
        return self.ground_truths[index]

    def load(self, dataset_dir):
        """Mock load method."""
        return {"mock": "data"}


class TestEvaluatorWithPassAt1:
    """Test Evaluator with PassAt1Scorer using fake data."""

    def test_calculate_score_perfect_accuracy(self):
        """Test calculate_score with 100% accuracy."""
        # Create mock dataset
        ground_truths = ["A", "B", "C", "D"]
        mock_dataset = MockDataset(ground_truths)

        # Create evaluator
        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        # Create temporary CSV file with extracted outputs
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "extracted_output"])
            writer.writerow([0, "A"])
            writer.writerow([1, "B"])
            writer.writerow([2, "C"])
            writer.writerow([3, "D"])
            temp_path = Path(f.name)

        try:
            # Calculate score
            score = evaluator.calculate_score(temp_path)

            # Should be 100% accuracy
            assert score == 1.0
        finally:
            temp_path.unlink()

    def test_calculate_score_zero_accuracy(self):
        """Test calculate_score with 0% accuracy."""
        ground_truths = ["A", "B", "C", "D"]
        mock_dataset = MockDataset(ground_truths)

        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "extracted_output"])
            # All wrong answers
            writer.writerow([0, "B"])  # Should be A
            writer.writerow([1, "C"])  # Should be B
            writer.writerow([2, "D"])  # Should be C
            writer.writerow([3, "A"])  # Should be D
            temp_path = Path(f.name)

        try:
            score = evaluator.calculate_score(temp_path)
            assert score == 0.0
        finally:
            temp_path.unlink()

    def test_calculate_score_partial_accuracy(self):
        """Test calculate_score with 50% accuracy."""
        ground_truths = ["A", "B", "C", "D"]
        mock_dataset = MockDataset(ground_truths)

        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "extracted_output"])
            writer.writerow([0, "A"])  # Correct
            writer.writerow([1, "C"])  # Wrong
            writer.writerow([2, "C"])  # Correct
            writer.writerow([3, "A"])  # Wrong
            temp_path = Path(f.name)

        try:
            score = evaluator.calculate_score(temp_path)
            assert score == 0.5
        finally:
            temp_path.unlink()

    def test_calculate_score_empty_file(self):
        """Test calculate_score with empty CSV (no data rows)."""
        mock_dataset = MockDataset([])

        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "extracted_output"])
            # No data rows
            temp_path = Path(f.name)

        try:
            score = evaluator.calculate_score(temp_path)
            assert score == 0.0  # Empty file returns 0.0
        finally:
            temp_path.unlink()

    def test_calculate_score_file_not_found(self):
        """Test calculate_score with non-existent file."""
        mock_dataset = MockDataset([])

        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        with pytest.raises(FileNotFoundError):
            evaluator.calculate_score(Path("/nonexistent/file.csv"))


class TestEvaluatorWithRougeScorer:
    """Test Evaluator with RougeScorer using fake data."""

    def test_calculate_score_returns_dict(self):
        """Test that ROUGE scorer returns dictionary of metrics."""
        ground_truths = [
            "the cat sat on the mat",
            "a dog was running",
            "the bird flew away",
        ]
        mock_dataset = MockDataset(ground_truths)

        evaluator = Evaluator(
            extractor=Mock(extract=lambda x: x),  # No-op extractor
            scorer=RougeScorer(use_stemmer=True),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "extracted_output"])
            writer.writerow([0, "the cat sat on the mat"])  # Perfect match
            writer.writerow([1, "a dog was running"])  # Perfect match
            writer.writerow([2, "the bird flew away"])  # Perfect match
            temp_path = Path(f.name)

        try:
            scores = evaluator.calculate_score(temp_path)

            # Should return dictionary
            assert isinstance(scores, dict)
            assert "rouge1" in scores
            assert "rouge2" in scores
            assert "rougeL" in scores

            # All perfect matches, should be close to 1.0
            assert scores["rouge1"] > 0.99
            assert scores["rouge2"] > 0.99
            assert scores["rougeL"] > 0.99
        finally:
            temp_path.unlink()

    def test_calculate_score_rouge_partial_match(self):
        """Test ROUGE scoring with partial matches."""
        ground_truths = [
            "the cat sat",
            "the dog ran",
        ]
        mock_dataset = MockDataset(ground_truths)

        evaluator = Evaluator(
            extractor=Mock(extract=lambda x: x),
            scorer=RougeScorer(use_stemmer=True),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "extracted_output"])
            writer.writerow([0, "the cat walked"])  # Partial match
            writer.writerow([1, "the dog jumped"])  # Partial match
            temp_path = Path(f.name)

        try:
            scores = evaluator.calculate_score(temp_path)

            # Should have intermediate scores
            assert 0.0 < scores["rouge1"] < 1.0
            assert 0.0 <= scores["rouge2"] < 1.0
            assert 0.0 < scores["rougeL"] < 1.0
        finally:
            temp_path.unlink()


class TestEvaluatorExtractOutputs:
    """Test Evaluator.extract_outputs method."""

    def test_extract_outputs_basic(self):
        """Test basic output extraction functionality."""
        import orjson

        mock_dataset = MockDataset([])
        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        # Create fake outputs file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
            f.write(orjson.dumps({"s_uuid": "uuid1", "output": "Answer: A"}))
            f.write(b"\n")
            f.write(
                orjson.dumps({"s_uuid": "uuid2", "output": "Answer: B"})
            )  # Fixed: use format that matches pattern
            f.write(b"\n")
            f.write(
                orjson.dumps({"s_uuid": "uuid3", "output": "Option C"})
            )  # Fixed: use format that matches pattern
            f.write(b"\n")
            outputs_path = Path(f.name)

        # Create temp file for extracted outputs
        extracted_path = Path(tempfile.mktemp(suffix=".csv"))

        uuid_map = {"uuid1": 0, "uuid2": 1, "uuid3": 2}

        try:
            evaluator.extract_outputs(
                outputs_file=outputs_path,
                save_to=extracted_path,
                uuid_map=uuid_map,
                num_samples=3,
            )

            # Read the extracted outputs
            with extracted_path.open("r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header == ["index", "extracted_output"]

                rows = list(reader)
                assert len(rows) == 3
                assert rows[0] == ["0", "A"]
                assert rows[1] == ["1", "B"]
                assert rows[2] == ["2", "C"]
        finally:
            outputs_path.unlink()
            if extracted_path.exists():
                extracted_path.unlink()

    def test_extract_outputs_with_no_match(self):
        """Test extraction when extractor finds no match."""
        import orjson

        mock_dataset = MockDataset([])
        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        # Create fake outputs file with no valid answers
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
            f.write(orjson.dumps({"s_uuid": "uuid1", "output": "I don't know"}))
            f.write(b"\n")
            outputs_path = Path(f.name)

        extracted_path = Path(tempfile.mktemp(suffix=".csv"))
        uuid_map = {"uuid1": 0}

        try:
            evaluator.extract_outputs(
                outputs_file=outputs_path,
                save_to=extracted_path,
                uuid_map=uuid_map,
                num_samples=1,
            )

            # Should write empty string when no match found
            with extracted_path.open("r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)
                assert rows[0] == ["0", ""]
        finally:
            outputs_path.unlink()
            if extracted_path.exists():
                extracted_path.unlink()

    def test_extract_outputs_with_offset(self):
        """Test extraction with row offset."""
        import orjson

        mock_dataset = MockDataset([])
        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        # Create fake outputs file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
            f.write(orjson.dumps({"s_uuid": "uuid1", "output": "A"}))
            f.write(b"\n")
            f.write(orjson.dumps({"s_uuid": "uuid2", "output": "B"}))
            f.write(b"\n")
            f.write(orjson.dumps({"s_uuid": "uuid3", "output": "C"}))
            f.write(b"\n")
            outputs_path = Path(f.name)

        extracted_path = Path(tempfile.mktemp(suffix=".csv"))
        uuid_map = {"uuid1": 0, "uuid2": 1, "uuid3": 2}

        try:
            # Skip first row, extract 2 rows
            evaluator.extract_outputs(
                outputs_file=outputs_path,
                save_to=extracted_path,
                uuid_map=uuid_map,
                num_samples=2,
                row_offset=1,
            )

            with extracted_path.open("r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)
                assert len(rows) == 2
                assert rows[0] == ["1", "B"]
                assert rows[1] == ["2", "C"]
        finally:
            outputs_path.unlink()
            if extracted_path.exists():
                extracted_path.unlink()


class TestEvaluatorEdgeCases:
    """Test edge cases and error handling."""

    def test_large_dataset(self):
        """Test with larger dataset."""
        # Create 100 samples - use letters for PassAt1Scorer
        ground_truths = ["A", "B", "C", "D"] * 25
        mock_dataset = MockDataset(ground_truths)

        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "extracted_output"])

            # Generate outputs with 75% accuracy
            # First 12 correct, last 4 wrong - pattern repeats
            outputs = [
                "A",
                "B",
                "C",
                "D",
                "A",
                "B",
                "C",
                "D",
                "A",
                "B",
                "C",
                "D",
                "B",
                "C",
                "D",
                "A",
            ]  # Last 4 are wrong
            outputs = (outputs * 7)[:100]  # 100 samples

            for i, output in enumerate(outputs):
                writer.writerow([i, output])

            temp_path = Path(f.name)

        try:
            score = evaluator.calculate_score(temp_path)
            # Should have 75% accuracy (12 correct out of 16, repeating)
            assert 0.7 < score < 0.8
        finally:
            temp_path.unlink()

    def test_mixed_empty_and_valid_outputs(self):
        """Test with mix of empty and valid outputs."""
        ground_truths = ["A", "B", "C", "D"]
        mock_dataset = MockDataset(ground_truths)

        evaluator = Evaluator(
            extractor=ABCDExtractor,
            scorer=PassAt1Scorer(),
            accuracy_dataset=mock_dataset,
            dataset_dir=Path("/fake/path"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "extracted_output"])
            writer.writerow([0, "A"])  # Correct
            writer.writerow([1, ""])  # Empty (wrong)
            writer.writerow([2, "C"])  # Correct
            writer.writerow([3, ""])  # Empty (wrong)
            temp_path = Path(f.name)

        try:
            score = evaluator.calculate_score(temp_path)
            assert score == 0.5  # 2 out of 4 correct
        finally:
            temp_path.unlink()
