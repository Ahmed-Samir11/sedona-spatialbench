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
Benchmark Engine Registry

This module provides registration and discovery of database engine implementations.
To add a new engine, import it here and add it to the ENGINES dictionary.
"""

from typing import Type
from benchmark.engine_base import BenchmarkEngine

# Import engine implementations
from benchmark.engines.duckdb_engine import DuckDBEngine
from benchmark.engines.sedona_engine import SedonaSparkEngine
from benchmark.engines.databricks_engine import DatabricksEngine
from benchmark.engines.geopandas_engine import GeopandasEngine
from benchmark.engines.polars_engine import SpatialPolarsEngine

# Registry of available engines
# Key: CLI argument name (lowercase)
# Value: Engine class
ENGINES: dict[str, Type[BenchmarkEngine]] = {
    "duckdb": DuckDBEngine,
    "sedona": SedonaSparkEngine,
    "databricks": DatabricksEngine,
    "geopandas": GeopandasEngine,
    "polars": SpatialPolarsEngine,
}


def get_engine(name: str) -> BenchmarkEngine:
    """Get an instance of the specified engine.

    Args:
        name: Engine name (case-insensitive)

    Returns:
        An instance of the requested engine

    Raises:
        ValueError: If engine name is not recognized
    """
    name_lower = name.lower()
    if name_lower not in ENGINES:
        available = ", ".join(sorted(ENGINES.keys()))
        raise ValueError(f"Unknown engine '{name}'. Available engines: {available}")

    return ENGINES[name_lower]()


def list_engines() -> list[str]:
    """Return list of available engine names."""
    return sorted(ENGINES.keys())


__all__ = [
    "ENGINES",
    "get_engine",
    "list_engines",
    "DuckDBEngine",
    "SedonaSparkEngine",
    "DatabricksEngine",
    "GeopandasEngine",
    "SpatialPolarsEngine",
]
