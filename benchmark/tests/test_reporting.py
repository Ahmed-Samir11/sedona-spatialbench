#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

"""
Tests for the reporting module.

These tests verify:
1. BenchmarkSummary correctly aggregates results
2. Statistical calculations (mean, median, std_dev) are accurate
3. Output formatting functions produce valid output
4. JSON/CSV export produces parseable files
"""

import json
import csv
import tempfile
from pathlib import Path

import pytest

from benchmark.engine_base import QueryResult
from benchmark.reporting import (
    BenchmarkSummary,
    save_json,
    save_csv,
)


class TestBenchmarkSummary:
    """Tests for the BenchmarkSummary class."""

    def test_init_with_required_fields(self):
        """BenchmarkSummary initializes with required fields."""
        summary = BenchmarkSummary(
            engine="duckdb",
            scale_factor=1.0,
            iterations=3,
        )

        assert summary.engine == "duckdb"
        assert summary.scale_factor == 1.0
        assert summary.iterations == 3
        assert summary.results == []

    def test_add_result(self):
        """add_result() appends QueryResult to results list."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        result = QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0)

        summary.add_result(result)

        assert len(summary.results) == 1
        assert summary.results[0] is result

    def test_total_queries_property(self):
        """total_queries returns correct count."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True))
        summary.add_result(QueryResult(query_name="q2", engine="test", success=True))

        assert summary.total_queries == 2

    def test_successful_queries_property(self):
        """successful_queries returns count of successful queries."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True))
        summary.add_result(QueryResult(query_name="q2", engine="test", success=False))
        summary.add_result(QueryResult(query_name="q3", engine="test", success=True))

        assert summary.successful_queries == 2

    def test_total_time_property(self):
        """total_time sums all query durations."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0))
        summary.add_result(QueryResult(query_name="q2", engine="test", success=True, duration_seconds=2.5))
        summary.add_result(QueryResult(query_name="q3", engine="test", success=True, duration_seconds=0.5))

        assert summary.total_time == pytest.approx(4.0)

    def test_total_time_ignores_none_durations(self):
        """total_time handles results with None duration."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0))
        summary.add_result(QueryResult(query_name="q2", engine="test", success=False))  # No duration

        assert summary.total_time == pytest.approx(1.0)

    def test_load_time_default_is_zero(self):
        """load_time defaults to 0.0."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)

        assert summary.load_time == 0.0

    def test_include_load_time_default_is_false(self):
        """include_load_time defaults to False."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)

        assert summary.include_load_time is False

    def test_get_total_time_without_load_time(self):
        """get_total_time() returns query time only when include_load_time is False."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=5.0))
        summary.load_time = 10.0
        summary.include_load_time = False

        assert summary.get_total_time() == pytest.approx(5.0)

    def test_get_total_time_with_load_time(self):
        """get_total_time() includes load time when include_load_time is True."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=5.0))
        summary.load_time = 10.0
        summary.include_load_time = True

        assert summary.get_total_time() == pytest.approx(15.0)


class TestBenchmarkSummaryAggregation:
    """Tests for BenchmarkSummary result aggregation."""

    def test_get_aggregated_results_calculates_mean(self):
        """Aggregated results calculate correct mean."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=3)

        # Add 3 iterations of q1 with durations 1.0, 2.0, 3.0
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0, iteration=1))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=2.0, iteration=2))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=3.0, iteration=3))

        agg = summary.get_aggregated_results()

        assert "q1" in agg
        assert agg["q1"]["mean"] == pytest.approx(2.0)

    def test_get_aggregated_results_calculates_median(self):
        """Aggregated results calculate correct median."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=3)

        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0, iteration=1))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=5.0, iteration=2))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=2.0, iteration=3))

        agg = summary.get_aggregated_results()

        # Median of [1.0, 5.0, 2.0] sorted = [1.0, 2.0, 5.0] -> median = 2.0
        assert agg["q1"]["median"] == pytest.approx(2.0)

    def test_get_aggregated_results_calculates_min_max(self):
        """Aggregated results calculate correct min and max."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=3)

        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.5, iteration=1))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=0.5, iteration=2))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=2.5, iteration=3))

        agg = summary.get_aggregated_results()

        assert agg["q1"]["min"] == pytest.approx(0.5)
        assert agg["q1"]["max"] == pytest.approx(2.5)

    def test_get_aggregated_results_calculates_success_rate(self):
        """Aggregated results calculate correct success rate."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=3)

        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0, iteration=1))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=False, iteration=2))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0, iteration=3))

        agg = summary.get_aggregated_results()

        # 2 out of 3 successful = 66.67%
        assert agg["q1"]["success_rate"] == pytest.approx(2 / 3)

    def test_get_aggregated_results_multiple_queries(self):
        """Aggregation works for multiple queries."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=2)

        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0, iteration=1))
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=2.0, iteration=2))
        summary.add_result(QueryResult(query_name="q2", engine="test", success=True, duration_seconds=3.0, iteration=1))
        summary.add_result(QueryResult(query_name="q2", engine="test", success=True, duration_seconds=4.0, iteration=2))

        agg = summary.get_aggregated_results()

        assert "q1" in agg
        assert "q2" in agg
        assert agg["q1"]["mean"] == pytest.approx(1.5)
        assert agg["q2"]["mean"] == pytest.approx(3.5)


class TestBenchmarkSummaryToDict:
    """Tests for BenchmarkSummary.to_dict() serialization."""

    def test_to_dict_includes_all_fields(self):
        """to_dict() includes all summary fields."""
        summary = BenchmarkSummary(engine="duckdb", scale_factor=1.0, iterations=3)
        summary.engine_version = "1.0.0"
        summary.add_result(QueryResult(query_name="q1", engine="duckdb", success=True, duration_seconds=1.0))

        d = summary.to_dict()

        assert d["engine"] == "duckdb"
        assert d["scale_factor"] == 1.0
        assert d["iterations"] == 3
        assert d["engine_version"] == "1.0.0"
        assert "raw_results" in d
        assert "summary" in d

    def test_to_dict_results_are_serialized(self):
        """to_dict() serializes individual results."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.5, row_count=100))

        d = summary.to_dict()

        assert len(d["raw_results"]) == 1
        assert d["raw_results"][0]["query_name"] == "q1"
        assert d["raw_results"][0]["duration_seconds"] == 1.5
        assert d["raw_results"][0]["row_count"] == 100

    def test_to_dict_includes_load_time_fields(self):
        """to_dict() includes load time information in summary."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=5.0))
        summary.load_time = 10.0
        summary.include_load_time = True

        d = summary.to_dict()

        assert d["summary"]["load_time_seconds"] == 10.0
        assert d["summary"]["query_time_seconds"] == 5.0
        assert d["summary"]["total_time_seconds"] == 15.0  # query + load
        assert d["summary"]["include_load_time"] is True


class TestSaveJson:
    """Tests for JSON export functionality."""

    def test_save_json_creates_valid_json(self):
        """save_json() creates a valid JSON file."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            save_json(summary, output_path)

            # Verify it's valid JSON
            with open(output_path) as f:
                data = json.load(f)

            assert data["engine"] == "test"
            assert len(data["raw_results"]) == 1
        finally:
            output_path.unlink()

    def test_save_json_creates_parent_dirs(self):
        """save_json() creates parent directories if needed."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "results.json"
            save_json(summary, output_path)

            assert output_path.exists()


class TestSaveCsv:
    """Tests for CSV export functionality."""

    def test_save_csv_creates_valid_csv(self):
        """save_csv() creates a valid CSV file."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True, duration_seconds=1.0, row_count=50))
        summary.add_result(QueryResult(query_name="q2", engine="test", success=True, duration_seconds=2.0, row_count=100))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            save_csv(summary, output_path)

            # Verify it's valid CSV
            with open(output_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["query_name"] == "q1"
            assert rows[1]["query_name"] == "q2"
        finally:
            output_path.unlink()

    def test_save_csv_has_header(self):
        """save_csv() includes a header row."""
        summary = BenchmarkSummary(engine="test", scale_factor=1.0, iterations=1)
        summary.add_result(QueryResult(query_name="q1", engine="test", success=True))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            save_csv(summary, output_path)

            with open(output_path) as f:
                header = f.readline().strip()

            assert "query_name" in header
            assert "engine" in header
            assert "success" in header
        finally:
            output_path.unlink()
