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
Apache Sedona (Spark) Benchmark Engine Implementation

This module provides a Sedona/Spark implementation of the BenchmarkEngine interface.
Sedona extends Apache Spark with spatial data types and functions for geospatial
analytics at scale.

Requirements:
    - pyspark
    - apache-sedona

Example:
    from benchmark.engines.sedona_engine import SedonaSparkEngine

    with SedonaSparkEngine() as engine:
        engine.load_data(Path("./data-sf1"), scale_factor=1.0)
        result = engine.run_query("q1", sql)
"""

import logging
from pathlib import Path
from typing import Any, Optional

from benchmark.engine_base import BenchmarkEngine, QueryResult, TimedExecution

logger = logging.getLogger(__name__)

# Standard SpatialBench tables
SPATIALBENCH_TABLES = ["trip", "customer", "driver", "vehicle", "building", "zone"]


class SedonaSparkEngine(BenchmarkEngine):
    """Apache Sedona (Spark) implementation of the benchmark engine.

    This engine runs benchmarks using Apache Spark with the Sedona spatial
    extension. It creates a local Spark session with Sedona context for
    executing spatial SQL queries.

    Attributes:
        _spark: SparkSession with Sedona context
        _data_loaded: Whether data has been loaded
        _app_name: Name of the Spark application
    """

    def __init__(self, app_name: str = "SpatialBench-Sedona") -> None:
        """Initialize Sedona Spark engine.

        Args:
            app_name: Name for the Spark application
        """
        self._spark: Optional[Any] = None
        self._data_loaded: bool = False
        self._app_name = app_name

    @property
    def name(self) -> str:
        """Return engine identifier."""
        return "sedona"

    @property
    def dialect(self) -> str:
        """Return SQL dialect for query generation."""
        return "SedonaSpark"

    def connect(self) -> None:
        """Create Spark session with Sedona context.

        Raises:
            ImportError: If pyspark or sedona packages are not installed
            RuntimeError: If Spark session creation fails
        """
        try:
            from sedona.spark import SedonaContext
        except ImportError as e:
            raise ImportError(
                "Apache Sedona is required for this engine. "
                "Install it with: pip install apache-sedona pyspark"
            ) from e

        logger.info(f"Creating Spark session with Sedona context: {self._app_name}")

        try:
            # Create Sedona context (this creates a SparkSession internally)
            self._spark = SedonaContext.create(
                SedonaContext.builder()
                .appName(self._app_name)
                .master("local[*]")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
                .getOrCreate()
            )
            logger.info("Sedona Spark session created successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to create Sedona Spark session: {e}") from e

    def load_data(self, data_dir: Path, scale_factor: float) -> None:
        """Load Parquet files as Spark temporary views.

        Args:
            data_dir: Directory containing Parquet files
            scale_factor: Scale factor (logged for reference)

        Raises:
            RuntimeError: If Spark session not established
            FileNotFoundError: If data directory doesn't exist
        """
        if self._spark is None:
            raise RuntimeError("Spark session not established. Call connect() first.")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        logger.info(f"Loading data from {data_dir} (scale factor: {scale_factor})")

        for table in SPATIALBENCH_TABLES:
            # Support both single file and partitioned directory structures
            parquet_path = data_dir / f"{table}.parquet"
            parquet_dir = data_dir / table

            if parquet_path.exists():
                source = str(parquet_path)
                logger.debug(f"Creating view '{table}' from {parquet_path}")
            elif parquet_dir.exists() and parquet_dir.is_dir():
                source = str(parquet_dir)
                logger.debug(f"Creating view '{table}' from {parquet_dir}")
            else:
                logger.warning(
                    f"Table '{table}' not found at {parquet_path} or {parquet_dir}"
                )
                continue

            try:
                df = self._spark.read.parquet(source)
                df.createOrReplaceTempView(table)
                count = df.count()
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
        if self._spark is None:
            return QueryResult(
                query_name=query_name,
                engine=self.name,
                success=False,
                error_message="Spark session not established",
            )

        logger.debug(f"Executing query {query_name}")

        with TimedExecution() as timer:
            try:
                # Execute query and force evaluation with collect()
                result_df = self._spark.sql(query_sql)
                rows = result_df.collect()
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
        """Stop Spark session and release resources."""
        if self._spark is not None:
            logger.info("Stopping Spark session")
            self._spark.stop()
            self._spark = None
            self._data_loaded = False

    def warmup(self) -> None:
        """Run a simple query to warm up Spark.

        This triggers JIT compilation and class loading before benchmarks.
        """
        if self._spark is None:
            return

        logger.debug("Running warmup query")
        try:
            self._spark.sql("SELECT 1").collect()
            # Warm up Sedona spatial functions
            self._spark.sql(
                "SELECT ST_GeomFromText('POINT(0 0)')"
            ).collect()
        except Exception as e:
            logger.warning(f"Warmup query failed: {e}")

    def get_version(self) -> str:
        """Return Spark and Sedona version strings."""
        if self._spark is None:
            return "unknown"

        try:
            spark_version = self._spark.version
            # Try to get Sedona version
            try:
                from sedona import __version__ as sedona_version
                return f"Spark {spark_version}, Sedona {sedona_version}"
            except ImportError:
                return f"Spark {spark_version}"
        except Exception:
            return "unknown"
