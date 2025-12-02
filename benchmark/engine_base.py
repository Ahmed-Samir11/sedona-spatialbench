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
Abstract Base Class for Benchmark Engines

This module defines the interface that all database engine implementations must follow.
It ensures consistency across different engines and makes adding new databases straightforward.

To implement a new engine:

    from benchmark.engine_base import BenchmarkEngine, QueryResult

    class MyNewEngine(BenchmarkEngine):
        '''Engine implementation for MyNewDB.'''

        @property
        def name(self) -> str:
            return "mynewdb"

        @property
        def dialect(self) -> str:
            return "MyNewDB"

        def connect(self) -> None:
            # Initialize connection to database
            self._conn = mynewdb.connect(...)

        def load_data(self, data_dir: Path, scale_factor: float) -> None:
            # Load Parquet files from data_dir as tables/views
            for table in ["trip", "customer", "driver", "vehicle", "building", "zone"]:
                parquet_path = data_dir / f"{table}.parquet"
                self._conn.execute(f"CREATE VIEW {table} AS SELECT * FROM '{parquet_path}'")

        def run_query(self, query_name: str, query_sql: str) -> QueryResult:
            # Execute query and return result with timing
            start = time.perf_counter()
            try:
                result = self._conn.execute(query_sql).fetchall()
                elapsed = time.perf_counter() - start
                return QueryResult(
                    query_name=query_name,
                    engine=self.name,
                    success=True,
                    duration_seconds=elapsed,
                    row_count=len(result)
                )
            except Exception as e:
                return QueryResult(
                    query_name=query_name,
                    engine=self.name,
                    success=False,
                    error_message=str(e)
                )

        def close(self) -> None:
            if self._conn:
                self._conn.close()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import time


@dataclass
class QueryResult:
    """Result of a single query execution.

    Attributes:
        query_name: Identifier for the query (e.g., "q1", "q2")
        engine: Name of the database engine used
        success: Whether the query completed successfully
        duration_seconds: Wall-clock time for query execution
        row_count: Number of rows returned (if successful)
        error_message: Error description (if failed)
        iteration: Which iteration this result is from (for multi-run benchmarks)
        metadata: Additional engine-specific metadata
    """

    query_name: str
    engine: str
    success: bool
    duration_seconds: float = 0.0
    row_count: int = 0
    error_message: Optional[str] = None
    iteration: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "query_name": self.query_name,
            "engine": self.engine,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "row_count": self.row_count,
            "error_message": self.error_message,
            "iteration": self.iteration,
            "metadata": self.metadata,
        }


class BenchmarkEngine(ABC):
    """Abstract base class for database benchmark engines.

    All database engine implementations must inherit from this class and
    implement all abstract methods. This ensures a consistent interface
    for the benchmark runner.

    The typical lifecycle is:
        1. engine = MyEngine()
        2. engine.connect()
        3. engine.load_data(data_dir, scale_factor)
        4. for query in queries:
               result = engine.run_query(query_name, query_sql)
        5. engine.close()

    Context manager support is provided for automatic cleanup:
        with MyEngine() as engine:
            engine.load_data(...)
            engine.run_query(...)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the short identifier for this engine (e.g., 'duckdb', 'sedona').

        This is used in CLI arguments and result reporting.
        """
        pass

    @property
    @abstractmethod
    def dialect(self) -> str:
        """Return the SQL dialect name matching print_queries.py.

        This must match one of the dialect names in print_queries.py:
        - "SedonaSpark"
        - "Databricks"
        - "DuckDB"
        - "SedonaDB"

        For non-SQL engines (e.g., Geopandas, Polars), this can return
        the engine name as a placeholder.
        """
        pass

    @property
    def uses_sql(self) -> bool:
        """Return whether this engine uses SQL for query execution.

        SQL engines execute queries via query_sql parameter in run_query().
        Non-SQL engines (like Geopandas, Polars) use query_name to dispatch
        to engine-specific function implementations and ignore query_sql.

        Default is True (SQL engine). Override to return False for
        DataFrame-based engines.
        """
        return True

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database.

        This method should initialize any necessary connections or resources.
        It is called once before loading data and running queries.

        Raises:
            ConnectionError: If unable to establish connection
        """
        pass

    @abstractmethod
    def load_data(self, data_dir: Path, scale_factor: float) -> None:
        """Load benchmark data from Parquet files.

        This method should create tables or views from the Parquet files
        in the data directory. The standard tables are:
        - trip (fact table with spatial pickup/dropoff columns)
        - customer
        - driver
        - vehicle
        - building (with spatial boundary column)
        - zone (with spatial boundary column)

        Args:
            data_dir: Directory containing Parquet files (*.parquet)
            scale_factor: The scale factor used to generate the data

        Raises:
            FileNotFoundError: If required Parquet files are missing
            RuntimeError: If data loading fails
        """
        pass

    @abstractmethod
    def run_query(self, query_name: str, query_sql: str, timeout: int = 1200) -> QueryResult:
        """Execute a benchmark query and measure performance.

        This method should:
        1. Execute the SQL query
        2. Measure wall-clock execution time
        3. Capture row count or error information
        4. Return a QueryResult with all relevant data

        Args:
            query_name: Identifier for the query (e.g., "q1")
            query_sql: The SQL query string to execute

        Returns:
            QueryResult containing timing and status information
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources and close connections.

        This method should release any database connections, temporary
        files, or other resources allocated during the benchmark run.
        """
        pass

    def warmup(self) -> None:
        """Optional warmup routine before running benchmarks.

        Override this method to implement engine-specific warmup logic,
        such as pre-compiling queries or warming caches.
        """
        pass

    def get_version(self) -> str:
        """Return the version string of the database engine.

        Override this method to provide version information for the
        benchmark report.
        """
        return "unknown"

    def __enter__(self) -> "BenchmarkEngine":
        """Context manager entry: connect to database."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit: close connection."""
        self.close()


class TimedExecution:
    """Context manager for timing code execution.

    Usage:
        with TimedExecution() as timer:
            # code to time
        print(f"Elapsed: {timer.elapsed:.3f}s")
    """

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "TimedExecution":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
