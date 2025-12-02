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
Integration tests for the SedonaDB engine.

These tests verify the SedonaDB engine implementation works end-to-end:
1. Connection and disconnection
2. Data loading from Parquet files
3. Query execution with timing
4. Warmup functionality

Note: These tests require sedonadb and apache-sedona to be installed.
"""

import tempfile
from pathlib import Path

import pytest

# Skip all tests if SedonaDB is not installed
sedonadb = pytest.importorskip("sedonadb")
pytest.importorskip("sedona.db")

from benchmark.engines.sedonadb_engine import SedonaDBEngine
from benchmark.engine_base import QueryResult


class TestSedonaDBEngineProperties:
    """Tests for SedonaDBEngine property values."""

    def test_name_is_sedonadb(self):
        """Engine name is 'sedonadb'."""
        engine = SedonaDBEngine()
        assert engine.name == "sedonadb"

    def test_dialect_is_sedonadb(self):
        """Engine dialect is 'SedonaDB'."""
        engine = SedonaDBEngine()
        assert engine.dialect == "SedonaDB"

    def test_uses_sql_is_true(self):
        """Engine uses SQL queries."""
        engine = SedonaDBEngine()
        assert engine.uses_sql is True


class TestSedonaDBEngineConnection:
    """Tests for SedonaDB connection management."""

    def test_connect_creates_context(self):
        """connect() creates a SedonaDB context."""
        engine = SedonaDBEngine()

        engine.connect()

        assert engine._ctx is not None
        engine.close()

    def test_close_releases_context(self):
        """close() releases the SedonaDB context."""
        engine = SedonaDBEngine()
        engine.connect()

        engine.close()

        assert engine._ctx is None

    def test_context_manager_connects_and_closes(self):
        """Context manager handles connection lifecycle."""
        engine = SedonaDBEngine()

        with engine:
            assert engine._ctx is not None

        assert engine._ctx is None

    def test_get_version_returns_version_string(self):
        """get_version() returns a version string."""
        engine = SedonaDBEngine()
        engine.connect()

        version = engine.get_version()

        assert version is not None
        assert isinstance(version, str)
        engine.close()


class TestSedonaDBEngineQueries:
    """Tests for SedonaDB query execution."""

    def test_run_query_returns_query_result(self):
        """run_query() returns a QueryResult."""
        engine = SedonaDBEngine()
        engine.connect()

        result = engine.run_query("test", "SELECT 1 AS x")

        assert isinstance(result, QueryResult)
        assert result.query_name == "test"
        assert result.engine == "sedonadb"
        engine.close()

    def test_run_query_success(self):
        """Successful query returns success=True with row count."""
        engine = SedonaDBEngine()
        engine.connect()

        result = engine.run_query("test", "SELECT 1 AS x UNION ALL SELECT 2 UNION ALL SELECT 3")

        assert result.success is True
        assert result.row_count == 3
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0
        engine.close()

    def test_run_query_failure(self):
        """Failed query returns success=False with error message."""
        engine = SedonaDBEngine()
        engine.connect()

        result = engine.run_query("test", "SELECT * FROM nonexistent_table_xyz")

        assert result.success is False
        assert result.error_message is not None
        engine.close()

    def test_run_query_without_connect_fails(self):
        """run_query() without connect() returns failure."""
        engine = SedonaDBEngine()

        result = engine.run_query("test", "SELECT 1")

        assert result.success is False
        assert "not established" in result.error_message.lower()


class TestSedonaDBEngineSpatial:
    """Tests for SedonaDB spatial functionality."""

    def test_spatial_function_works(self):
        """Spatial functions work in SedonaDB."""
        engine = SedonaDBEngine()
        engine.connect()

        result = engine.run_query("spatial_test", "SELECT ST_Point(0, 0) AS geom")

        assert result.success is True
        assert result.row_count == 1
        engine.close()

    def test_st_geomfromtext_works(self):
        """ST_GeomFromText works for WKT parsing."""
        engine = SedonaDBEngine()
        engine.connect()

        result = engine.run_query(
            "wkt_test",
            "SELECT ST_GeomFromText('POINT(1 2)') AS geom"
        )

        assert result.success is True
        assert result.row_count == 1
        engine.close()

    def test_warmup_runs_without_error(self):
        """warmup() executes without raising."""
        engine = SedonaDBEngine()
        engine.connect()

        # Should not raise
        engine.warmup()

        engine.close()


class TestSedonaDBEngineDataLoading:
    """Tests for SedonaDB data loading functionality."""

    @pytest.fixture
    def sample_parquet_dir(self):
        """Create a temporary directory with sample Parquet files."""
        pyarrow = pytest.importorskip("pyarrow")
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create a simple trip.parquet file
            table = pa.table({
                "t_tripkey": [1, 2, 3],
                "t_fare": [10.0, 20.0, 30.0],
            })
            pq.write_table(table, tmppath / "trip.parquet")

            # Create customer.parquet
            table = pa.table({
                "c_custkey": [1, 2],
                "c_name": ["Alice", "Bob"],
            })
            pq.write_table(table, tmppath / "customer.parquet")

            yield tmppath

    def test_load_data_creates_views(self, sample_parquet_dir):
        """load_data() creates views for Parquet files."""
        engine = SedonaDBEngine()
        engine.connect()

        engine.load_data(sample_parquet_dir, scale_factor=1.0)

        # Query the loaded view
        result = engine.run_query("test", "SELECT COUNT(*) FROM trip")
        assert result.success is True
        assert result.row_count == 1  # One row with count

        engine.close()

    def test_load_data_missing_dir_raises(self):
        """load_data() raises FileNotFoundError for missing directory."""
        engine = SedonaDBEngine()
        engine.connect()

        with pytest.raises(FileNotFoundError):
            engine.load_data(Path("/nonexistent/path/that/does/not/exist"), scale_factor=1.0)

        engine.close()

    def test_load_data_without_connect_raises(self):
        """load_data() without connect() raises RuntimeError."""
        engine = SedonaDBEngine()

        with pytest.raises(RuntimeError, match="not established"):
            engine.load_data(Path("."), scale_factor=1.0)


class TestSedonaDBEngineRegistry:
    """Tests for SedonaDB engine registration."""

    def test_sedonadb_in_engines_registry(self):
        """sedonadb is registered in ENGINES."""
        from benchmark.engines import ENGINES

        assert "sedonadb" in ENGINES

    def test_get_engine_returns_sedonadb(self):
        """get_engine('sedonadb') returns SedonaDBEngine."""
        from benchmark.engines import get_engine

        engine = get_engine("sedonadb")

        assert isinstance(engine, SedonaDBEngine)

    def test_list_engines_includes_sedonadb(self):
        """list_engines() includes 'sedonadb'."""
        from benchmark.engines import list_engines

        engines = list_engines()

        assert "sedonadb" in engines
