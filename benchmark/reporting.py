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
Benchmark Result Reporting

This module provides utilities for formatting and outputting benchmark results.
"""

import json
import csv
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

from benchmark.engine_base import QueryResult


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark results.

    Attributes:
        engine: Engine name
        total_queries: Number of queries run
        successful_queries: Number of queries that succeeded
        failed_queries: Number of queries that failed
        total_time: Total time for all queries
        results: List of individual query results
        engine_version: Version of the database engine
        scale_factor: Scale factor of the benchmark data
        iterations: Number of iterations per query
        load_time: Time taken to load data (seconds)
        include_load_time: Whether to include load time in total
    """

    engine: str
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_time: float = 0.0
    results: list[QueryResult] = field(default_factory=list)
    engine_version: str = "unknown"
    scale_factor: float = 1.0
    iterations: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    load_time: float = 0.0
    include_load_time: bool = False

    def add_result(self, result: QueryResult) -> None:
        """Add a query result to the summary."""
        self.results.append(result)
        self.total_queries += 1
        if result.success:
            self.successful_queries += 1
            self.total_time += result.duration_seconds
        else:
            self.failed_queries += 1

    def get_aggregated_results(self) -> dict[str, dict]:
        """Get results aggregated by query name (for multi-iteration runs).

        Returns:
            Dictionary mapping query names to aggregated statistics:
            {
                "q1": {
                    "min": 0.1,
                    "max": 0.2,
                    "mean": 0.15,
                    "median": 0.15,
                    "std_dev": 0.05,
                    "iterations": 3,
                    "success_rate": 1.0
                }
            }
        """
        # Group results by query name
        by_query: dict[str, list[QueryResult]] = {}
        for result in self.results:
            if result.query_name not in by_query:
                by_query[result.query_name] = []
            by_query[result.query_name].append(result)

        # Calculate statistics for each query
        aggregated = {}
        for query_name, results in by_query.items():
            successful = [r for r in results if r.success]
            times = [r.duration_seconds for r in successful]

            if times:
                aggregated[query_name] = {
                    "min": min(times),
                    "max": max(times),
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                    "iterations": len(results),
                    "successful_iterations": len(successful),
                    "success_rate": len(successful) / len(results),
                }
            else:
                # All iterations failed
                aggregated[query_name] = {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "std_dev": None,
                    "iterations": len(results),
                    "successful_iterations": 0,
                    "success_rate": 0.0,
                    "errors": [r.error_message for r in results if r.error_message],
                }

        return aggregated

    def get_total_time(self) -> float:
        """Get total time, optionally including data load time.

        Returns:
            Total time in seconds (query time + load time if include_load_time is True)
        """
        if self.include_load_time:
            return self.total_time + self.load_time
        return self.total_time

    def to_dict(self) -> dict:
        """Convert summary to dictionary for JSON serialization."""
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "scale_factor": self.scale_factor,
            "iterations": self.iterations,
            "timestamp": self.timestamp,
            "summary": {
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "failed_queries": self.failed_queries,
                "total_time_seconds": self.get_total_time(),
                "query_time_seconds": self.total_time,
                "load_time_seconds": self.load_time,
                "include_load_time": self.include_load_time,
            },
            "aggregated_results": self.get_aggregated_results(),
            "raw_results": [r.to_dict() for r in self.results],
        }


def print_results_table(
    results: list[QueryResult],
    file: TextIO | None = None,
    show_iteration: bool = False,
) -> None:
    """Print benchmark results as an ASCII table.

    Args:
        results: List of query results to display
        file: Output file (defaults to stdout)
        show_iteration: Whether to show iteration column
    """
    import sys

    if file is None:
        file = sys.stdout

    # Define columns
    if show_iteration:
        headers = ["Query", "Engine", "Iter", "Duration (s)", "Rows", "Status"]
        widths = [8, 12, 4, 14, 10, 8]
    else:
        headers = ["Query", "Engine", "Duration (s)", "Rows", "Status"]
        widths = [8, 12, 14, 10, 8]

    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    separator = "-+-".join("-" * w for w in widths)

    print(separator, file=file)
    print(header_line, file=file)
    print(separator, file=file)

    # Print rows
    for result in results:
        status = "OK" if result.success else "FAIL"
        duration = f"{result.duration_seconds:.4f}" if result.success else "-"
        rows = str(result.row_count) if result.success else "-"

        if show_iteration:
            row = [
                result.query_name.ljust(widths[0]),
                result.engine.ljust(widths[1]),
                str(result.iteration).ljust(widths[2]),
                duration.rjust(widths[3]),
                rows.rjust(widths[4]),
                status.ljust(widths[5]),
            ]
        else:
            row = [
                result.query_name.ljust(widths[0]),
                result.engine.ljust(widths[1]),
                duration.rjust(widths[2]),
                rows.rjust(widths[3]),
                status.ljust(widths[4]),
            ]

        print(" | ".join(row), file=file)

    print(separator, file=file)


def print_aggregated_table(
    summary: BenchmarkSummary,
    file: TextIO | None = None,
) -> None:
    """Print aggregated benchmark results (for multi-iteration runs).

    Args:
        summary: Benchmark summary with results
        file: Output file (defaults to stdout)
    """
    import sys

    if file is None:
        file = sys.stdout

    aggregated = summary.get_aggregated_results()

    headers = ["Query", "Engine", "Mean (s)", "Median (s)", "Min (s)", "Max (s)", "StdDev", "Success"]
    widths = [8, 12, 10, 10, 10, 10, 8, 8]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    separator = "-+-".join("-" * w for w in widths)

    print(separator, file=file)
    print(header_line, file=file)
    print(separator, file=file)

    for query_name in sorted(aggregated.keys(), key=lambda x: int(x[1:])):
        stats = aggregated[query_name]
        success_rate = f"{stats['success_rate']*100:.0f}%"

        if stats["mean"] is not None:
            row = [
                query_name.ljust(widths[0]),
                summary.engine.ljust(widths[1]),
                f"{stats['mean']:.4f}".rjust(widths[2]),
                f"{stats['median']:.4f}".rjust(widths[3]),
                f"{stats['min']:.4f}".rjust(widths[4]),
                f"{stats['max']:.4f}".rjust(widths[5]),
                f"{stats['std_dev']:.4f}".rjust(widths[6]),
                success_rate.rjust(widths[7]),
            ]
        else:
            row = [
                query_name.ljust(widths[0]),
                summary.engine.ljust(widths[1]),
                "-".rjust(widths[2]),
                "-".rjust(widths[3]),
                "-".rjust(widths[4]),
                "-".rjust(widths[5]),
                "-".rjust(widths[6]),
                success_rate.rjust(widths[7]),
            ]

        print(" | ".join(row), file=file)

    print(separator, file=file)


def print_comparison_table(
    summaries: list[BenchmarkSummary],
    file: TextIO | None = None,
) -> None:
    """Print side-by-side comparison of multiple engines.

    Args:
        summaries: List of benchmark summaries to compare
        file: Output file (defaults to stdout)
    """
    import sys

    if file is None:
        file = sys.stdout

    if not summaries:
        print("No results to compare", file=file)
        return

    # Get all query names
    all_queries = set()
    for summary in summaries:
        all_queries.update(summary.get_aggregated_results().keys())
    sorted_queries = sorted(all_queries, key=lambda x: int(x[1:]))

    # Build header
    engine_names = [s.engine for s in summaries]
    headers = ["Query"] + [f"{name} (s)" for name in engine_names]
    widths = [8] + [12] * len(engine_names)

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    separator = "-+-".join("-" * w for w in widths)

    print("\n" + "=" * len(separator), file=file)
    print("COMPARISON: Mean Query Times", file=file)
    print("=" * len(separator), file=file)
    print(header_line, file=file)
    print(separator, file=file)

    # Print each query row
    for query_name in sorted_queries:
        row = [query_name.ljust(widths[0])]

        for i, summary in enumerate(summaries):
            agg = summary.get_aggregated_results()
            if query_name in agg and agg[query_name]["mean"] is not None:
                row.append(f"{agg[query_name]['mean']:.4f}".rjust(widths[i + 1]))
            else:
                row.append("-".rjust(widths[i + 1]))

        print(" | ".join(row), file=file)

    print(separator, file=file)

    # Print totals
    row = ["TOTAL".ljust(widths[0])]
    for i, summary in enumerate(summaries):
        row.append(f"{summary.total_time:.4f}".rjust(widths[i + 1]))
    print(" | ".join(row), file=file)
    print(separator, file=file)


def save_json(summary: BenchmarkSummary, output_path: Path) -> None:
    """Save benchmark results to JSON file.

    Args:
        summary: Benchmark summary to save
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)


def save_csv(summary: BenchmarkSummary, output_path: Path) -> None:
    """Save benchmark results to CSV file.

    Args:
        summary: Benchmark summary to save
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregated = summary.get_aggregated_results()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_name",
            "engine",
            "mean_seconds",
            "median_seconds",
            "min_seconds",
            "max_seconds",
            "std_dev",
            "iterations",
            "success_rate",
        ])

        for query_name in sorted(aggregated.keys(), key=lambda x: int(x[1:])):
            stats = aggregated[query_name]
            writer.writerow([
                query_name,
                summary.engine,
                stats["mean"],
                stats["median"],
                stats["min"],
                stats["max"],
                stats["std_dev"],
                stats["iterations"],
                stats["success_rate"],
            ])
