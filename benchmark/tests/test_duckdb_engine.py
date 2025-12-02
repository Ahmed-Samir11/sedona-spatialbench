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
Integration tests for the DuckDB engine.

These tests verify the DuckDB engine implementation works end-to-end:
1. Connection and disconnection
2. Data loading from Parquet files
3. Query execution with timing
4. Warmup functionality

Note: These tests require DuckDB to be installed.
"""

import tempfile
from pathlib import Path

import pytest

# Skip all tests if DuckDB is not installed
duckdb = pytest.importorskip("duckdb")

from benchmark.engines.duckdb_engine import DuckDBEngine
from benchmark.engine_base import QueryResult


class TestDuckDBEngineProperties:
    """Tests for DuckDBEngine property values."""

    def test_name_is_duckdb(self):
        """Engine name is 'duckdb'."""
        engine = DuckDBEngine()
        assert engine.name == "duckdb"

    def test_dialect_is_duckdb(self):
        """Engine dialect is 'DuckDB'."""
        engine = DuckDBEngine()
        assert engine.dialect == "DuckDB"

    def test_uses_sql_is_true(self):
        """Engine uses SQL queries."""
        engine = DuckDBEngine()
        assert engine.uses_sql is True


class TestDuckDBEngineConnection:
    """Tests for DuckDB connection management."""

    def test_connect_creates_connection(self):
        """connect() creates a DuckDB connection."""
        engine = DuckDBEngine()

        engine.connect()

        assert engine._conn is not None
        engine.close()

    def test_close_releases_connection(self):
        """close() releases the DuckDB connection."""
        engine = DuckDBEngine()
        engine.connect()

        engine.close()

        assert engine._conn is None

    def test_context_manager_connects_and_closes(self):
        """Context manager handles connection lifecycle."""
        engine = DuckDBEngine()

        with engine:
            assert engine._conn is not None

        assert engine._conn is None

    def test_get_version_returns_duckdb_version(self):
        """get_version() returns DuckDB version string."""
        engine = DuckDBEngine()
        engine.connect()

        version = engine.get_version()

        assert "duckdb" in version.lower()
        engine.close()


class TestDuckDBEngineQueries:
    """Tests for DuckDB query execution."""

    def test_run_query_returns_query_result(self):
        """run_query() returns a QueryResult."""
        engine = DuckDBEngine()
        engine.connect()

        result = engine.run_query("test", "SELECT 1 AS x")

        assert isinstance(result, QueryResult)
        assert result.query_name == "test"
        assert result.engine == "duckdb"
        engine.close()

    def test_run_query_success(self):
        """Successful query returns success=True with row count."""
        engine = DuckDBEngine()
        engine.connect()

        result = engine.run_query("test", "SELECT * FROM (VALUES (1), (2), (3)) AS t(x)")

        assert result.success is True
        assert result.row_count == 3
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0
        engine.close()

    def test_run_query_failure(self):
        """Failed query returns success=False with error message."""
        engine = DuckDBEngine()
        engine.connect()

        result = engine.run_query("test", "SELECT * FROM nonexistent_table")

        assert result.success is False
        assert result.error_message is not None
        assert "nonexistent_table" in result.error_message.lower()
        engine.close()

    def test_run_query_without_connect_fails(self):
        """run_query() without connect() returns failure."""
        engine = DuckDBEngine()

        result = engine.run_query("test", "SELECT 1")

        assert result.success is False
        assert "not established" in result.error_message.lower()


class TestDuckDBEngineSpatial:
    """Tests for DuckDB spatial functionality."""

    def test_spatial_extension_loaded(self):
        """Spatial extension is loaded on connect."""
        engine = DuckDBEngine()
        engine.connect()

        # This should work if spatial extension is loaded
        result = engine.run_query("spatial_test", "SELECT ST_Point(0, 0)")

        assert result.success is True
        engine.close()

    def test_warmup_runs_spatial_query(self):
        """warmup() exercises spatial functions."""
        engine = DuckDBEngine()
        engine.connect()

        # Should not raise
        engine.warmup()

        engine.close()


class TestDuckDBEngineDataLoading:
    """Tests for DuckDB data loading functionality."""

    @pytest.fixture
    def sample_parquet_dir(self):
        """Create a temporary directory with sample Parquet files."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create a simple trip.parquet file
            table = pa.table({
                "trip_id": [1, 2, 3],
                "fare": [10.0, 20.0, 30.0],
            })
            pq.write_table(table, tmppath / "trip.parquet")

            # Create customer.parquet
            table = pa.table({
                "customer_id": [1, 2],
                "name": ["Alice", "Bob"],
            })
            pq.write_table(table, tmppath / "customer.parquet")

            yield tmppath

    def test_load_data_creates_views(self, sample_parquet_dir):
        """load_data() creates views for Parquet files."""
        pyarrow = pytest.importorskip("pyarrow")

        engine = DuckDBEngine()
        engine.connect()

        engine.load_data(sample_parquet_dir, scale_factor=1.0)

        # Query the loaded view
        result = engine.run_query("test", "SELECT COUNT(*) FROM trip")
        assert result.success is True
        assert result.row_count == 1  # One row with count

        engine.close()

    def test_load_data_missing_dir_raises(self):
        """load_data() raises FileNotFoundError for missing directory."""
        engine = DuckDBEngine()
        engine.connect()

        with pytest.raises(FileNotFoundError):
            engine.load_data(Path("/nonexistent/path"), scale_factor=1.0)

        engine.close()

    def test_load_data_without_connect_raises(self):
        """load_data() without connect() raises RuntimeError."""
        engine = DuckDBEngine()

        with pytest.raises(RuntimeError, match="not established"):
            engine.load_data(Path("."), scale_factor=1.0)
