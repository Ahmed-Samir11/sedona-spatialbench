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
Databricks SQL Benchmark Engine Implementation

This module provides a Databricks SQL implementation of the BenchmarkEngine
interface. It connects to a Databricks SQL warehouse via the SQL connector
and executes spatial queries using Databricks' built-in spatial functions.

Requirements:
    - databricks-sql-connector

Configuration via environment variables:
    - DATABRICKS_SERVER_HOSTNAME: Workspace hostname (e.g., xxx.cloud.databricks.com)
    - DATABRICKS_HTTP_PATH: SQL warehouse HTTP path
    - DATABRICKS_ACCESS_TOKEN: Personal access token or OAuth token

Example:
    export DATABRICKS_SERVER_HOSTNAME="your-workspace.cloud.databricks.com"
    export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/abc123"
    export DATABRICKS_ACCESS_TOKEN="dapi..."

    from benchmark.engines.databricks_engine import DatabricksEngine

    with DatabricksEngine() as engine:
        engine.load_data(Path("./data-sf1"), scale_factor=1.0)
        result = engine.run_query("q1", sql)
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from benchmark.engine_base import BenchmarkEngine, QueryResult, TimedExecution

logger = logging.getLogger(__name__)

# Standard SpatialBench tables
SPATIALBENCH_TABLES = ["trip", "customer", "driver", "vehicle", "building", "zone"]

# Environment variable names
ENV_SERVER_HOSTNAME = "DATABRICKS_SERVER_HOSTNAME"
ENV_HTTP_PATH = "DATABRICKS_HTTP_PATH"
ENV_ACCESS_TOKEN = "DATABRICKS_ACCESS_TOKEN"
ENV_CATALOG = "DATABRICKS_CATALOG"
ENV_SCHEMA = "DATABRICKS_SCHEMA"


class DatabricksEngine(BenchmarkEngine):
    """Databricks SQL implementation of the benchmark engine.

    This engine connects to a Databricks SQL warehouse and executes queries
    using the databricks-sql-connector. Credentials are read from environment
    variables.

    Attributes:
        _connection: Databricks SQL connection
        _catalog: Unity Catalog name
        _schema: Schema name for benchmark tables
    """

    def __init__(
        self,
        server_hostname: Optional[str] = None,
        http_path: Optional[str] = None,
        access_token: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        """Initialize Databricks engine.

        Args:
            server_hostname: Databricks workspace hostname
            http_path: SQL warehouse HTTP path
            access_token: Personal access token or OAuth token
            catalog: Unity Catalog name (default: from env or "main")
            schema: Schema name (default: from env or "default")
        """
        self._server_hostname = server_hostname or os.environ.get(ENV_SERVER_HOSTNAME)
        self._http_path = http_path or os.environ.get(ENV_HTTP_PATH)
        self._access_token = access_token or os.environ.get(ENV_ACCESS_TOKEN)
        self._catalog = catalog or os.environ.get(ENV_CATALOG, "main")
        self._schema = schema or os.environ.get(ENV_SCHEMA, "spatialbench")
        self._connection: Optional[Any] = None
        self._data_loaded: bool = False

    @property
    def name(self) -> str:
        """Return engine identifier."""
        return "databricks"

    @property
    def dialect(self) -> str:
        """Return SQL dialect for query generation."""
        return "Databricks"

    def _validate_config(self) -> None:
        """Validate that required configuration is present.

        Raises:
            ValueError: If required environment variables are missing
        """
        missing = []
        if not self._server_hostname:
            missing.append(ENV_SERVER_HOSTNAME)
        if not self._http_path:
            missing.append(ENV_HTTP_PATH)
        if not self._access_token:
            missing.append(ENV_ACCESS_TOKEN)

        if missing:
            raise ValueError(
                f"Missing required Databricks configuration: {', '.join(missing)}. "
                "Set these environment variables or pass them to the constructor."
            )

    def connect(self) -> None:
        """Establish connection to Databricks SQL warehouse.

        Raises:
            ImportError: If databricks-sql-connector is not installed
            ValueError: If required configuration is missing
            RuntimeError: If connection fails
        """
        try:
            from databricks import sql as databricks_sql
        except ImportError as e:
            raise ImportError(
                "databricks-sql-connector is required for this engine. "
                "Install it with: pip install databricks-sql-connector"
            ) from e

        self._validate_config()

        logger.info(f"Connecting to Databricks: {self._server_hostname}")

        try:
            self._connection = databricks_sql.connect(
                server_hostname=self._server_hostname,
                http_path=self._http_path,
                access_token=self._access_token,
            )
            logger.info("Databricks connection established")

            # Set catalog and schema context
            with self._connection.cursor() as cursor:
                cursor.execute(f"USE CATALOG {self._catalog}")
                cursor.execute(f"USE SCHEMA {self._schema}")
                logger.info(f"Using catalog: {self._catalog}, schema: {self._schema}")

        except Exception as e:
            raise RuntimeError(f"Failed to connect to Databricks: {e}") from e

    def load_data(self, data_dir: Path, scale_factor: float) -> None:
        """Verify tables exist in Databricks.

        Note: Data must be pre-loaded to Databricks as Delta tables.
        This method verifies the expected tables exist and logs row counts.

        Args:
            data_dir: Directory containing Parquet files (for logging only)
            scale_factor: Scale factor (logged for reference)

        Raises:
            RuntimeError: If connection not established or tables not found
        """
        if self._connection is None:
            raise RuntimeError("Connection not established. Call connect() first.")

        logger.info(
            f"Verifying tables in {self._catalog}.{self._schema} "
            f"(scale factor: {scale_factor})"
        )

        missing_tables = []
        with self._connection.cursor() as cursor:
            for table in SPATIALBENCH_TABLES:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    result = cursor.fetchone()
                    count = result[0] if result else 0
                    logger.info(f"Table '{table}': {count:,} rows")
                except Exception as e:
                    logger.warning(f"Table '{table}' not found: {e}")
                    missing_tables.append(table)

        if missing_tables:
            logger.warning(
                f"Missing tables: {missing_tables}. "
                "Ensure data is loaded to Databricks Delta tables."
            )

        self._data_loaded = True
        logger.info("Table verification complete")

    def run_query(self, query_name: str, query_sql: str) -> QueryResult:
        """Execute a query and measure performance.

        Args:
            query_name: Query identifier (e.g., "q1")
            query_sql: SQL query to execute

        Returns:
            QueryResult with timing and status information
        """
        if self._connection is None:
            return QueryResult(
                query_name=query_name,
                engine=self.name,
                success=False,
                error_message="Connection not established",
            )

        logger.debug(f"Executing query {query_name}")

        with TimedExecution() as timer:
            try:
                with self._connection.cursor() as cursor:
                    cursor.execute(query_sql)
                    rows = cursor.fetchall()
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
        """Close Databricks connection."""
        if self._connection is not None:
            logger.info("Closing Databricks connection")
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            self._connection = None
            self._data_loaded = False

    def get_version(self) -> str:
        """Return Databricks connector version."""
        try:
            from databricks.sql import __version__
            return f"databricks-sql-connector {__version__}"
        except (ImportError, AttributeError):
            return "unknown"
