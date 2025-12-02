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
Tests for BenchmarkEngine base class and QueryResult dataclass.

These tests verify:
1. QueryResult correctly stores and reports execution results
2. TimedExecution context manager accurately measures elapsed time
3. BenchmarkEngine ABC enforces the required interface
4. Context manager protocol (__enter__/__exit__) works correctly
"""

import time
import pytest
from abc import ABC
from pathlib import Path
from unittest.mock import MagicMock

from benchmark.engine_base import BenchmarkEngine, QueryResult, TimedExecution


class TestQueryResult:
    """Tests for the QueryResult dataclass."""

    def test_successful_query_result(self):
        """QueryResult correctly represents a successful query execution."""
        result = QueryResult(
            query_name="q1",
            engine="test_engine",
            success=True,
            duration_seconds=1.234,
            row_count=100,
            iteration=1,
        )

        assert result.query_name == "q1"
        assert result.engine == "test_engine"
        assert result.success is True
        assert result.duration_seconds == 1.234
        assert result.row_count == 100
        assert result.iteration == 1
        assert result.error_message is None

    def test_failed_query_result(self):
        """QueryResult correctly represents a failed query execution."""
        result = QueryResult(
            query_name="q2",
            engine="test_engine",
            success=False,
            duration_seconds=0.5,
            error_message="Connection timeout",
        )

        assert result.query_name == "q2"
        assert result.success is False
        assert result.error_message == "Connection timeout"
        assert result.row_count == 0  # Defaults to 0

    def test_query_result_defaults(self):
        """QueryResult has sensible defaults for optional fields."""
        result = QueryResult(
            query_name="q1",
            engine="test",
            success=True,
        )

        assert result.duration_seconds == 0.0  # Default
        assert result.row_count == 0  # Default
        assert result.error_message is None
        assert result.iteration == 1  # Default

    def test_query_result_to_dict(self):
        """QueryResult.to_dict() produces correct dictionary representation."""
        result = QueryResult(
            query_name="q1",
            engine="duckdb",
            success=True,
            duration_seconds=2.5,
            row_count=50,
            iteration=3,
        )

        d = result.to_dict()

        assert d["query_name"] == "q1"
        assert d["engine"] == "duckdb"
        assert d["success"] is True
        assert d["duration_seconds"] == 2.5
        assert d["row_count"] == 50
        assert d["iteration"] == 3


class TestTimedExecution:
    """Tests for the TimedExecution context manager."""

    def test_measures_elapsed_time(self):
        """TimedExecution accurately measures elapsed time."""
        with TimedExecution() as timer:
            time.sleep(0.1)  # Sleep 100ms

        # Should be at least 100ms, but allow some tolerance
        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.5  # Should not be excessively long

    def test_elapsed_available_after_exit(self):
        """Elapsed time is available after context manager exits."""
        timer = TimedExecution()

        with timer:
            time.sleep(0.05)

        # Elapsed should be frozen after exit
        elapsed1 = timer.elapsed
        time.sleep(0.05)
        elapsed2 = timer.elapsed

        assert elapsed1 == elapsed2  # Should not change after exit

    def test_elapsed_zero_before_exit(self):
        """Elapsed time is zero until context exits (set in __exit__)."""
        with TimedExecution() as timer:
            time.sleep(0.05)
            # Elapsed is only calculated after __exit__
            pass

        # After exit, elapsed should be set
        assert timer.elapsed > 0


class TestBenchmarkEngineABC:
    """Tests for the BenchmarkEngine abstract base class."""

    def test_is_abstract_class(self):
        """BenchmarkEngine cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BenchmarkEngine()

    def test_required_abstract_methods(self):
        """BenchmarkEngine requires specific abstract methods."""
        # Get all abstract methods
        abstract_methods = BenchmarkEngine.__abstractmethods__

        required_methods = {"name", "dialect", "connect", "load_data", "run_query", "close"}
        assert required_methods.issubset(abstract_methods)

    def test_subclass_must_implement_all_methods(self):
        """Subclass must implement all abstract methods."""

        class IncompleteEngine(BenchmarkEngine):
            @property
            def name(self):
                return "incomplete"

            @property
            def dialect(self):
                return "test"

            # Missing: connect, load_data, run_query, close

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteEngine()

    def test_valid_subclass_can_be_instantiated(self):
        """A complete subclass can be instantiated."""

        class CompleteEngine(BenchmarkEngine):
            @property
            def name(self):
                return "complete"

            @property
            def dialect(self):
                return "test"

            def connect(self):
                pass

            def load_data(self, data_dir, scale_factor):
                pass

            def run_query(self, query_name, query_sql):
                return QueryResult(query_name=query_name, engine=self.name, success=True)

            def close(self):
                pass

        engine = CompleteEngine()
        assert engine.name == "complete"
        assert engine.dialect == "test"

    def test_uses_sql_defaults_to_true(self):
        """uses_sql property defaults to True for SQL-based engines."""

        class SQLEngine(BenchmarkEngine):
            @property
            def name(self):
                return "sql_engine"

            @property
            def dialect(self):
                return "test"

            def connect(self):
                pass

            def load_data(self, data_dir, scale_factor):
                pass

            def run_query(self, query_name, query_sql):
                return QueryResult(query_name=query_name, engine=self.name, success=True)

            def close(self):
                pass

        engine = SQLEngine()
        assert engine.uses_sql is True

    def test_uses_sql_can_be_overridden(self):
        """uses_sql property can be overridden for function-based engines."""

        class FunctionEngine(BenchmarkEngine):
            @property
            def name(self):
                return "function_engine"

            @property
            def dialect(self):
                return "test"

            @property
            def uses_sql(self):
                return False

            def connect(self):
                pass

            def load_data(self, data_dir, scale_factor):
                pass

            def run_query(self, query_name, query_sql):
                return QueryResult(query_name=query_name, engine=self.name, success=True)

            def close(self):
                pass

        engine = FunctionEngine()
        assert engine.uses_sql is False


class TestBenchmarkEngineContextManager:
    """Tests for BenchmarkEngine context manager protocol."""

    def test_context_manager_calls_connect_and_close(self):
        """Context manager calls connect() on enter and close() on exit."""

        class TrackingEngine(BenchmarkEngine):
            def __init__(self):
                self.connect_called = False
                self.close_called = False

            @property
            def name(self):
                return "tracking"

            @property
            def dialect(self):
                return "test"

            def connect(self):
                self.connect_called = True

            def load_data(self, data_dir, scale_factor):
                pass

            def run_query(self, query_name, query_sql):
                return QueryResult(query_name=query_name, engine=self.name, success=True)

            def close(self):
                self.close_called = True

        engine = TrackingEngine()

        assert engine.connect_called is False
        assert engine.close_called is False

        with engine:
            assert engine.connect_called is True
            assert engine.close_called is False

        assert engine.close_called is True

    def test_context_manager_closes_on_exception(self):
        """Context manager calls close() even when exception occurs."""

        class FailingEngine(BenchmarkEngine):
            def __init__(self):
                self.close_called = False

            @property
            def name(self):
                return "failing"

            @property
            def dialect(self):
                return "test"

            def connect(self):
                pass

            def load_data(self, data_dir, scale_factor):
                raise RuntimeError("Load failed")

            def run_query(self, query_name, query_sql):
                return QueryResult(query_name=query_name, engine=self.name, success=True)

            def close(self):
                self.close_called = True

        engine = FailingEngine()

        with pytest.raises(RuntimeError, match="Load failed"):
            with engine:
                engine.load_data(Path("."), 1.0)

        assert engine.close_called is True


class TestBenchmarkEngineDefaultMethods:
    """Tests for default method implementations in BenchmarkEngine."""

    def test_warmup_is_no_op_by_default(self):
        """Default warmup() does nothing (no exception)."""

        class MinimalEngine(BenchmarkEngine):
            @property
            def name(self):
                return "minimal"

            @property
            def dialect(self):
                return "test"

            def connect(self):
                pass

            def load_data(self, data_dir, scale_factor):
                pass

            def run_query(self, query_name, query_sql):
                return QueryResult(query_name=query_name, engine=self.name, success=True)

            def close(self):
                pass

        engine = MinimalEngine()
        # Should not raise
        engine.warmup()

    def test_get_version_returns_unknown_by_default(self):
        """Default get_version() returns 'unknown'."""

        class MinimalEngine(BenchmarkEngine):
            @property
            def name(self):
                return "minimal"

            @property
            def dialect(self):
                return "test"

            def connect(self):
                pass

            def load_data(self, data_dir, scale_factor):
                pass

            def run_query(self, query_name, query_sql):
                return QueryResult(query_name=query_name, engine=self.name, success=True)

            def close(self):
                pass

        engine = MinimalEngine()
        assert engine.get_version() == "unknown"
