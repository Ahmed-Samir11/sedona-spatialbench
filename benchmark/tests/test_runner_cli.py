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
Integration tests for the CLI runner.

These tests verify the runner.py CLI works correctly:
1. Command-line argument parsing
2. Engine listing
3. Query listing
4. Help output
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Path to runner.py
RUNNER_PATH = Path(__file__).parent.parent.parent / "runner.py"


def run_runner(*args: str) -> subprocess.CompletedProcess:
    """Run runner.py with given arguments and return result."""
    return subprocess.run(
        [sys.executable, str(RUNNER_PATH), *args],
        capture_output=True,
        text=True,
    )


class TestRunnerCLI:
    """Tests for runner.py CLI behavior."""

    def test_help_shows_usage(self):
        """--help shows usage information."""
        result = run_runner("--help")

        assert result.returncode == 0
        assert "SpatialBench Benchmark Runner" in result.stdout
        assert "--engine" in result.stdout
        assert "--data-dir" in result.stdout

    def test_list_engines_shows_all_engines(self):
        """--list-engines shows all registered engines."""
        result = run_runner("--list-engines")

        assert result.returncode == 0
        assert "duckdb" in result.stdout
        assert "sedona" in result.stdout
        assert "databricks" in result.stdout
        assert "geopandas" in result.stdout
        assert "polars" in result.stdout

    def test_list_engines_shows_sql_type(self):
        """--list-engines shows SQL vs function-based type."""
        result = run_runner("--list-engines")

        assert result.returncode == 0
        assert "SQL" in result.stdout
        assert "function-based" in result.stdout

    def test_list_queries_shows_all_queries(self):
        """--list-queries shows all 12 benchmark queries."""
        result = run_runner("--list-queries")

        assert result.returncode == 0
        for i in range(1, 13):
            assert f"q{i}" in result.stdout

    def test_missing_engine_shows_error(self):
        """Missing --engine argument shows error."""
        result = run_runner("--data-dir", ".")

        assert result.returncode != 0
        assert "engine" in result.stderr.lower() or "required" in result.stderr.lower()

    def test_missing_data_dir_shows_error(self):
        """Missing --data-dir argument shows error."""
        result = run_runner("--engine", "duckdb")

        assert result.returncode != 0
        assert "data-dir" in result.stderr.lower() or "required" in result.stderr.lower()

    def test_invalid_engine_shows_error(self):
        """Invalid engine name shows helpful error."""
        result = run_runner("--engine", "invalid_engine", "--data-dir", ".")

        assert result.returncode != 0
        # Should mention the invalid engine or list available ones
        output = result.stdout + result.stderr
        assert "invalid" in output.lower() or "unknown" in output.lower() or "available" in output.lower()

    def test_nonexistent_data_dir_shows_error(self):
        """Non-existent data directory shows error."""
        result = run_runner("--engine", "duckdb", "--data-dir", "/nonexistent/path/12345")

        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert "not found" in output.lower() or "exist" in output.lower()


class TestRunnerOptions:
    """Tests for runner.py optional arguments."""

    def test_iterations_default_is_3(self):
        """Default iterations is 3."""
        result = run_runner("--help")

        assert "default: 3" in result.stdout.lower()

    def test_scale_factor_default_is_1(self):
        """Default scale factor is 1.0."""
        result = run_runner("--help")

        assert "default: 1.0" in result.stdout

    def test_verbose_flag_accepted(self):
        """--verbose flag is accepted."""
        result = run_runner("--help")

        assert "--verbose" in result.stdout or "-v" in result.stdout

    def test_output_formats_accepted(self):
        """--output-format accepts json and csv."""
        result = run_runner("--help")

        assert "json" in result.stdout
        assert "csv" in result.stdout

    def test_include_load_time_flag_accepted(self):
        """--include-load-time flag is accepted and documented."""
        result = run_runner("--help")

        assert "--include-load-time" in result.stdout
        assert "data loading time" in result.stdout.lower() or "load time" in result.stdout.lower()
