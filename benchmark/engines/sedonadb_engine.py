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

"""SedonaDB engine implementation for SpatialBench.

This engine uses the `sedona.db` Python package which provides
a SedonaContext with methods:
  - read_parquet(path) -> DataFrame
  - view(name, df) -> register DataFrame as temp view
  - sql(query) -> DataFrame result

The DataFrame has .collect() to get rows as list of dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from benchmark.engine_base import BenchmarkEngine, QueryResult, TimedExecution

logger = logging.getLogger(__name__)

# Standard SpatialBench tables
SPATIALBENCH_TABLES = ["trip", "customer", "driver", "vehicle", "building", "zone"]


class SedonaDBEngine(BenchmarkEngine):
    """SedonaDB engine using the ``sedona.db`` Python package.

    The SedonaContext exposes:
      - read_parquet(path) to load parquet files as DataFrames
      - view(name, df) to register DataFrames as temp views
      - sql(query) to execute SQL and return a DataFrame
    """

    def __init__(self) -> None:
        self._ctx: Optional[Any] = None
        self._data_loaded: bool = False

    @property
    def name(self) -> str:
        return "sedonadb"

    @property
    def dialect(self) -> str:
        return "SedonaDB"

    def connect(self) -> None:
        try:
            import sedona.db as sdb
        except ImportError as e:
            raise ImportError(
                "The 'apache-sedona' and 'sedonadb' packages are required. "
                "Install with: pip install apache-sedona sedonadb"
            ) from e

        try:
            self._ctx = sdb.connect()
        except Exception as e:
            raise RuntimeError(f"Failed to create SedonaDB context: {e}") from e

        logger.info("SedonaDB context established")

    def load_data(self, data_dir: Path, scale_factor: float) -> None:
        if self._ctx is None:
            raise RuntimeError("Context not established. Call connect() first.")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        logger.info(f"Loading data from {data_dir} (scale factor: {scale_factor})")

        for table in SPATIALBENCH_TABLES:
            parquet_path = data_dir / f"{table}.parquet"
            parquet_dir = data_dir / table

            if parquet_path.exists():
                source = str(parquet_path.resolve())
            elif parquet_dir.exists() and parquet_dir.is_dir():
                source = str(parquet_dir.resolve())
            else:
                logger.warning(f"Table '{table}' not found at {parquet_path} or {parquet_dir}")
                continue

            try:
                # read_parquet returns a DataFrame, then register as view via to_view()
                df = self._ctx.read_parquet(source)
                df.to_view(table, overwrite=True)
                logger.info(f"Registered table '{table}' from {source}")
            except Exception as e:
                logger.warning(f"Failed to register table '{table}': {e}")

        self._data_loaded = True

    def run_query(self, query_name: str, query_sql: str) -> QueryResult:
        if self._ctx is None:
            return QueryResult(
                query_name=query_name,
                engine=self.name,
                success=False,
                error_message="Context not established",
            )

        logger.debug(f"Executing query {query_name}")
        with TimedExecution() as timer:
            try:
                result_df = self._ctx.sql(query_sql)
                # to_arrow_table() returns a PyArrow Table with len() for row count
                arrow_table = result_df.to_arrow_table()
                row_count = len(arrow_table)
            except Exception as e:
                logger.error(f"Query {query_name} failed: {e}")
                return QueryResult(
                    query_name=query_name,
                    engine=self.name,
                    success=False,
                    duration_seconds=timer.elapsed,
                    error_message=str(e),
                )

        return QueryResult(
            query_name=query_name,
            engine=self.name,
            success=True,
            duration_seconds=timer.elapsed,
            row_count=row_count,
        )

    def close(self) -> None:
        # SedonaContext doesn't require explicit close
        self._ctx = None
        self._data_loaded = False

    def warmup(self) -> None:
        if self._ctx is None:
            return

        try:
            self._ctx.sql("SELECT 1").collect()
        except Exception:
            pass

    def get_version(self) -> str:
        try:
            import sedonadb

            return getattr(sedonadb, "__version__", "unknown")
        except Exception:
            return "unknown"
