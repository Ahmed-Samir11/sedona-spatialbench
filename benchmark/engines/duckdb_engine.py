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
DuckDB Benchmark Engine Implementation

This module provides a DuckDB implementation of the BenchmarkEngine interface.
DuckDB is used as the reference implementation due to its:
- In-memory operation (no external server required)
- Native Parquet support
- Built-in spatial extension
"""

import logging
from pathlib import Path
from typing import Any, Optional

from benchmark.engine_base import BenchmarkEngine, QueryResult, TimedExecution

logger = logging.getLogger(__name__)

# Standard SpatialBench tables
SPATIALBENCH_TABLES = ["trip", "customer", "driver", "vehicle", "building", "zone"]


class DuckDBEngine(BenchmarkEngine):
    """DuckDB implementation of the benchmark engine.

    This engine runs benchmarks using an in-memory DuckDB database with
    the spatial extension enabled. Data is loaded directly from Parquet
    files as views for efficient querying.

    Attributes:
        _conn: DuckDB connection object
        _data_loaded: Whether data has been loaded
    """

    def __init__(self) -> None:
        """Initialize DuckDB engine."""
        self._conn: Optional[Any] = None
        self._data_loaded: bool = False

    @property
    def name(self) -> str:
        """Return engine identifier."""
        return "duckdb"

    @property
    def dialect(self) -> str:
        """Return SQL dialect for query generation."""
        return "DuckDB"

    def connect(self) -> None:
        """Create in-memory DuckDB connection with spatial extension.

        Raises:
            ImportError: If duckdb package is not installed
            RuntimeError: If spatial extension cannot be loaded
        """
        try:
            import duckdb
        except ImportError as e:
            raise ImportError(
                "DuckDB is required for this engine. "
                "Install it with: pip install duckdb"
            ) from e

        logger.info("Creating in-memory DuckDB connection")
        self._conn = duckdb.connect(":memory:")

        # Install and load spatial extension
        logger.info("Loading DuckDB spatial extension")
        try:
            self._conn.execute("INSTALL spatial;")
            self._conn.execute("LOAD spatial;")
        except Exception as e:
            raise RuntimeError(f"Failed to load DuckDB spatial extension: {e}") from e

        # Disable external file cache for consistency
        self._conn.execute("SET enable_external_file_cache = false")

        logger.info("DuckDB connection established with spatial extension")

    def load_data(self, data_dir: Path, scale_factor: float) -> None:
        """Load Parquet files from data directory as DuckDB views.

        Creates views for each SpatialBench table, allowing DuckDB to
        read directly from Parquet files without loading into memory.

        Args:
            data_dir: Directory containing Parquet files
            scale_factor: Scale factor (logged for reference)

        Raises:
            RuntimeError: If connection not established
            FileNotFoundError: If data directory doesn't exist
        """
        if self._conn is None:
            raise RuntimeError("Connection not established. Call connect() first.")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        logger.info(f"Loading data from {data_dir} (scale factor: {scale_factor})")

        for table in SPATIALBENCH_TABLES:
            # Support both single file and partitioned directory structures
            parquet_path = data_dir / f"{table}.parquet"
            parquet_dir = data_dir / table

            if parquet_path.exists():
                # Single Parquet file
                source = str(parquet_path)
                logger.debug(f"Creating view '{table}' from {parquet_path}")
            elif parquet_dir.exists() and parquet_dir.is_dir():
                # Partitioned Parquet directory (e.g., trip/*.parquet)
                source = str(parquet_dir / "*.parquet")
                logger.debug(f"Creating view '{table}' from {parquet_dir}")
            else:
                logger.warning(
                    f"Table '{table}' not found at {parquet_path} or {parquet_dir}"
                )
                continue

            try:
                # Use read_parquet for flexible Parquet loading
                self._conn.execute(
                    f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{source}')"
                )
                # Log row count for verification
                count = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[
                    0
                ]
                logger.info(f"Loaded table '{table}': {count:,} rows")
            except Exception as e:
                logger.error(f"Failed to load table '{table}': {e}")
                raise RuntimeError(f"Failed to load table '{table}': {e}") from e

        self._data_loaded = True
        logger.info("All tables loaded successfully")

    def run_query(self, query_name: str, query_sql: str) -> QueryResult:
        """Execute a query and measure performance.

        Args:
            query_name: Query identifier (e.g., "q1")
            query_sql: SQL query to execute

        Returns:
            QueryResult with timing and status information
        """
        if self._conn is None:
            return QueryResult(
                query_name=query_name,
                engine=self.name,
                success=False,
                error_message="Connection not established",
            )

        logger.debug(f"Executing query {query_name}")

        with TimedExecution() as timer:
            try:
                result = self._conn.execute(query_sql)
                rows = result.fetchall()
                row_count = len(rows)
            except Exception as e:
                logger.error(f"Query {query_name} failed: {e}")
                return QueryResult(
                    query_name=query_name,
                    engine=self.name,
                    success=False,
                    duration_seconds=timer.elapsed,
                    error_message=str(e),
                )

        logger.debug(
            f"Query {query_name} completed: {row_count} rows in {timer.elapsed:.3f}s"
        )

        return QueryResult(
            query_name=query_name,
            engine=self.name,
            success=True,
            duration_seconds=timer.elapsed,
            row_count=row_count,
        )

    def close(self) -> None:
        """Close DuckDB connection and release resources."""
        if self._conn is not None:
            logger.info("Closing DuckDB connection")
            self._conn.close()
            self._conn = None
            self._data_loaded = False

    def warmup(self) -> None:
        """Run a simple query to warm up the database.

        This helps ensure consistent timing by triggering any lazy
        initialization before the benchmark starts.
        """
        if self._conn is None:
            return

        logger.debug("Running warmup query")
        try:
            self._conn.execute("SELECT 1").fetchall()
            # Also warm up spatial extension
            self._conn.execute(
                "SELECT ST_GeomFromText('POINT(0 0)')"
            ).fetchall()
        except Exception as e:
            logger.warning(f"Warmup query failed: {e}")

    def get_version(self) -> str:
        """Return DuckDB version string."""
        if self._conn is None:
            return "unknown"

        try:
            result = self._conn.execute("SELECT version()").fetchone()
            return result[0] if result else "unknown"
        except Exception:
            return "unknown"
