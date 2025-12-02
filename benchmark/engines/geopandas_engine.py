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
GeoPandas Benchmark Engine Implementation

This module provides a GeoPandas implementation of the BenchmarkEngine interface.
Unlike SQL-based engines, this engine executes queries as Python functions that
operate on GeoDataFrames directly.

The query functions are defined in the top-level geopandas.py file and are
imported and called dynamically based on query name.

Requirements:
    - geopandas
    - pandas
    - shapely

Example:
    from benchmark.engines.geopandas_engine import GeopandasEngine

    with GeopandasEngine() as engine:
        engine.load_data(Path("./data-sf1"), scale_factor=1.0)
        result = engine.run_query("q1", "")  # SQL ignored for this engine
"""

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from benchmark.engine_base import BenchmarkEngine, QueryResult, TimedExecution

logger = logging.getLogger(__name__)

# Standard SpatialBench tables
SPATIALBENCH_TABLES = ["trip", "customer", "driver", "vehicle", "building", "zone"]


class GeopandasEngine(BenchmarkEngine):
    """GeoPandas implementation of the benchmark engine.

    This engine executes benchmark queries using GeoPandas DataFrames rather
    than SQL. Query implementations are loaded from the geopandas.py module
    in the project root.

    Attributes:
        _data_paths: Mapping of table names to Parquet file paths
        _query_funcs: Mapping of query names to query functions
    """

    def __init__(self) -> None:
        """Initialize GeoPandas engine."""
        self._data_paths: Dict[str, str] = {}
        self._query_funcs: Dict[str, Callable] = {}
        self._geopandas_module: Optional[Any] = None

    @property
    def name(self) -> str:
        """Return engine identifier."""
        return "geopandas"

    @property
    def dialect(self) -> str:
        """Return dialect (not used for non-SQL engines)."""
        return "Geopandas"

    @property
    def uses_sql(self) -> bool:
        """Indicate this engine does not use SQL."""
        return False

    def _load_query_module(self) -> None:
        """Load the geopandas.py query module.

        Raises:
            ImportError: If geopandas.py cannot be loaded
        """
        if self._geopandas_module is not None:
            return

        try:
            import geopandas as gpd  # noqa: F401 - Verify geopandas is installed
        except ImportError as e:
            raise ImportError(
                "GeoPandas is required for this engine. "
                "Install it with: pip install geopandas"
            ) from e

        # Import the geopandas.py module from project root
        # We need to import it as a module, not a package
        try:
            import importlib.util

            # Find geopandas.py in project root (parent of benchmark/)
            project_root = Path(__file__).parent.parent.parent
            geopandas_file = project_root / "geopandas.py"

            if not geopandas_file.exists():
                raise ImportError(f"Query module not found: {geopandas_file}")

            spec = importlib.util.spec_from_file_location(
                "geopandas_queries", geopandas_file
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {geopandas_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["geopandas_queries"] = module
            spec.loader.exec_module(module)
            self._geopandas_module = module

            # Collect query functions (q1, q2, ..., q12)
            for i in range(1, 13):
                func_name = f"q{i}"
                if hasattr(module, func_name):
                    self._query_funcs[func_name] = getattr(module, func_name)
                    logger.debug(f"Loaded query function: {func_name}")

            logger.info(f"Loaded {len(self._query_funcs)} query functions from geopandas.py")

        except Exception as e:
            raise ImportError(f"Failed to load geopandas query module: {e}") from e

    def connect(self) -> None:
        """Initialize engine by loading query module.

        Raises:
            ImportError: If required packages are not installed
        """
        logger.info("Initializing GeoPandas engine")
        self._load_query_module()
        logger.info("GeoPandas engine initialized")

    def load_data(self, data_dir: Path, scale_factor: float) -> None:
        """Build data path mapping for query functions.

        Args:
            data_dir: Directory containing Parquet files
            scale_factor: Scale factor (logged for reference)

        Raises:
            FileNotFoundError: If data directory doesn't exist
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        logger.info(f"Building data paths from {data_dir} (scale factor: {scale_factor})")

        self._data_paths.clear()

        for table in SPATIALBENCH_TABLES:
            # Support both single file and partitioned directory structures
            parquet_path = data_dir / f"{table}.parquet"
            parquet_dir = data_dir / table

            if parquet_path.exists():
                self._data_paths[table] = str(parquet_path)
                logger.debug(f"Data path '{table}': {parquet_path}")
            elif parquet_dir.exists() and parquet_dir.is_dir():
                self._data_paths[table] = str(parquet_dir)
                logger.debug(f"Data path '{table}': {parquet_dir}")
            else:
                logger.warning(
                    f"Table '{table}' not found at {parquet_path} or {parquet_dir}"
                )

        logger.info(f"Data paths configured for {len(self._data_paths)} tables")

    def run_query(self, query_name: str, query_sql: str) -> QueryResult:
        """Execute a query function and measure performance.

        Args:
            query_name: Query identifier (e.g., "q1")
            query_sql: Ignored for this engine

        Returns:
            QueryResult with timing and status information
        """
        if query_name not in self._query_funcs:
            return QueryResult(
                query_name=query_name,
                engine=self.name,
                success=False,
                error_message=f"Query function '{query_name}' not found",
            )

        if not self._data_paths:
            return QueryResult(
                query_name=query_name,
                engine=self.name,
                success=False,
                error_message="No data paths configured. Call load_data() first.",
            )

        logger.debug(f"Executing query {query_name}")
        query_func = self._query_funcs[query_name]

        with TimedExecution() as timer:
            try:
                # Call the query function with data paths
                result_df = query_func(self._data_paths)
                row_count = len(result_df) if result_df is not None else 0
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
        """Clean up engine resources."""
        logger.info("Closing GeoPandas engine")
        self._data_paths.clear()
        self._query_funcs.clear()
        self._geopandas_module = None

    def get_available_queries(self) -> list[str]:
        """Return list of available query names."""
        return sorted(self._query_funcs.keys())

    def get_version(self) -> str:
        """Return GeoPandas version string."""
        try:
            import geopandas
            return f"geopandas {geopandas.__version__}"
        except (ImportError, AttributeError):
            return "unknown"
