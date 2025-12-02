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
Query Provider for SpatialBench Benchmarks

This module provides access to the benchmark queries defined in print_queries.py.
It dynamically loads queries for the appropriate dialect based on the engine.
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path for importing print_queries
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import query classes from print_queries.py
from print_queries import (
    SpatialBenchBenchmark,
    DatabricksSpatialBenchBenchmark,
    DuckDBSpatialBenchBenchmark,
    SedonaDBSpatialBenchBenchmark,
)

# Mapping from dialect names to query classes
QUERY_CLASSES: dict[str, type] = {
    "SedonaSpark": SpatialBenchBenchmark,
    "Databricks": DatabricksSpatialBenchBenchmark,
    "DuckDB": DuckDBSpatialBenchBenchmark,
    "SedonaDB": SedonaDBSpatialBenchBenchmark,
}


def get_queries(dialect: str) -> dict[str, str]:
    """Get benchmark queries for a specific SQL dialect.

    Args:
        dialect: SQL dialect name (e.g., "DuckDB", "SedonaSpark")

    Returns:
        Dictionary mapping query names (e.g., "q1") to SQL strings

    Raises:
        ValueError: If dialect is not supported
    """
    if dialect not in QUERY_CLASSES:
        available = ", ".join(sorted(QUERY_CLASSES.keys()))
        raise ValueError(f"Unknown dialect '{dialect}'. Available: {available}")

    benchmark = QUERY_CLASSES[dialect]()
    return benchmark.queries()


def get_query(dialect: str, query_name: str) -> Optional[str]:
    """Get a specific query for a dialect.

    Args:
        dialect: SQL dialect name
        query_name: Query identifier (e.g., "q1")

    Returns:
        SQL query string, or None if query not found
    """
    queries = get_queries(dialect)
    return queries.get(query_name)


def list_queries(dialect: str) -> list[str]:
    """List available query names for a dialect.

    Args:
        dialect: SQL dialect name

    Returns:
        Sorted list of query names
    """
    queries = get_queries(dialect)
    return sorted(queries.keys(), key=lambda x: int(x[1:]))


def list_dialects() -> list[str]:
    """Return list of supported SQL dialects."""
    return sorted(QUERY_CLASSES.keys())


def get_query_count() -> int:
    """Return the number of benchmark queries.

    Uses SedonaSpark as the reference since all dialects should
    have the same number of queries.
    """
    return len(get_queries("SedonaSpark"))
