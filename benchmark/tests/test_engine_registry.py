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
Tests for the engine registry module.

These tests verify:
1. All engines are properly registered
2. get_engine() returns correct engine instances
3. list_engines() returns all available engines
4. Unknown engine names raise appropriate errors
"""

import pytest

from benchmark.engines import (
    ENGINES,
    get_engine,
    list_engines,
    DuckDBEngine,
    SedonaSparkEngine,
    SedonaDBEngine,
    DatabricksEngine,
    GeopandasEngine,
    SpatialPolarsEngine,
)
from benchmark.engine_base import BenchmarkEngine


class TestEngineRegistry:
    """Tests for the ENGINES registry dictionary."""

    def test_all_expected_engines_registered(self):
        """All expected engines are registered in ENGINES."""
        expected_engines = {"duckdb", "sedona", "sedonadb", "databricks", "geopandas", "polars"}
        registered_engines = set(ENGINES.keys())

        assert expected_engines == registered_engines

    def test_registry_maps_to_correct_classes(self):
        """Registry maps engine names to correct classes."""
        assert ENGINES["duckdb"] is DuckDBEngine
        assert ENGINES["sedona"] is SedonaSparkEngine
        assert ENGINES["sedonadb"] is SedonaDBEngine
        assert ENGINES["databricks"] is DatabricksEngine
        assert ENGINES["geopandas"] is GeopandasEngine
        assert ENGINES["polars"] is SpatialPolarsEngine

    def test_all_registered_classes_are_benchmark_engines(self):
        """All registered classes are subclasses of BenchmarkEngine."""
        for name, engine_class in ENGINES.items():
            assert issubclass(engine_class, BenchmarkEngine), (
                f"Engine '{name}' is not a BenchmarkEngine subclass"
            )


class TestGetEngine:
    """Tests for the get_engine() function."""

    def test_get_engine_returns_instance(self):
        """get_engine() returns an instance of the requested engine."""
        engine = get_engine("duckdb")

        assert isinstance(engine, DuckDBEngine)
        assert isinstance(engine, BenchmarkEngine)

    def test_get_engine_case_insensitive(self):
        """get_engine() is case-insensitive."""
        engine1 = get_engine("duckdb")
        engine2 = get_engine("DuckDB")
        engine3 = get_engine("DUCKDB")

        assert type(engine1) is type(engine2) is type(engine3)

    def test_get_engine_unknown_raises_value_error(self):
        """get_engine() raises ValueError for unknown engine names."""
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("nonexistent_engine")

    def test_get_engine_error_lists_available(self):
        """ValueError message lists available engines."""
        with pytest.raises(ValueError) as exc_info:
            get_engine("invalid")

        error_msg = str(exc_info.value)
        assert "duckdb" in error_msg
        assert "sedona" in error_msg

    def test_get_engine_returns_new_instance_each_time(self):
        """get_engine() returns a new instance on each call."""
        engine1 = get_engine("duckdb")
        engine2 = get_engine("duckdb")

        assert engine1 is not engine2


class TestListEngines:
    """Tests for the list_engines() function."""

    def test_list_engines_returns_all_engines(self):
        """list_engines() returns all registered engine names."""
        engines = list_engines()

        assert "duckdb" in engines
        assert "sedona" in engines
        assert "databricks" in engines
        assert "geopandas" in engines
        assert "polars" in engines

    def test_list_engines_returns_sorted_list(self):
        """list_engines() returns engines in sorted order."""
        engines = list_engines()

        assert engines == sorted(engines)

    def test_list_engines_returns_list(self):
        """list_engines() returns a list type."""
        engines = list_engines()

        assert isinstance(engines, list)

    def test_list_engines_count_matches_registry(self):
        """list_engines() returns same count as ENGINES."""
        engines = list_engines()

        assert len(engines) == len(ENGINES)


class TestEngineProperties:
    """Tests for engine property values."""

    @pytest.mark.parametrize("engine_name,expected_dialect,expected_uses_sql", [
        ("duckdb", "DuckDB", True),
        ("sedona", "SedonaSpark", True),
        ("databricks", "Databricks", True),
        ("geopandas", "Geopandas", False),
        ("polars", "Polars", False),
    ])
    def test_engine_dialect_and_uses_sql(self, engine_name, expected_dialect, expected_uses_sql):
        """Each engine has correct dialect and uses_sql values."""
        engine = get_engine(engine_name)

        assert engine.dialect == expected_dialect
        assert engine.uses_sql == expected_uses_sql

    def test_engine_names_match_registry_keys(self):
        """Each engine's name property matches its registry key."""
        for registry_name in list_engines():
            engine = get_engine(registry_name)
            assert engine.name == registry_name, (
                f"Engine registered as '{registry_name}' has name '{engine.name}'"
            )
