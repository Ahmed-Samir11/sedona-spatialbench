#!/usr/bin/env python3
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
SpatialBench Benchmark Runner

A CLI tool for executing SpatialBench benchmark queries against various
database engines and measuring performance.

Usage Examples:
    # Run all queries against DuckDB with data at scale factor 1
    python runner.py --engine duckdb --data-dir ./data-sf1

    # Run 5 iterations of each query and save results
    python runner.py --engine duckdb --data-dir ./data-sf1 --iterations 5 --output results.json

    # Run specific queries only
    python runner.py --engine duckdb --data-dir ./data-sf1 --queries q1,q2,q3

    # Compare all available engines
    python runner.py --engine all --data-dir ./data-sf1 --output comparison.json

    # List available engines
    python runner.py --list-engines
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark.engines import get_engine, list_engines, ENGINES
from benchmark.engine_base import BenchmarkEngine, QueryResult
from benchmark.query_provider import get_queries, list_queries, get_query_count
from benchmark.reporting import (
    BenchmarkSummary,
    print_results_table,
    print_aggregated_table,
    print_comparison_table,
    save_json,
    save_csv,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_benchmark(
    engine: BenchmarkEngine,
    data_dir: Path,
    scale_factor: float,
    iterations: int = 1,
    queries: Optional[list[str]] = None,
    warmup: bool = True,
) -> BenchmarkSummary:
    """Run benchmark queries against an engine.

    Args:
        engine: The database engine to use
        data_dir: Directory containing Parquet data files
        scale_factor: Scale factor of the benchmark data
        iterations: Number of times to run each query
        queries: Specific queries to run (None = all)
        warmup: Whether to run warmup before benchmarks

    Returns:
        BenchmarkSummary with all results
    """
    summary = BenchmarkSummary(
        engine=engine.name,
        scale_factor=scale_factor,
        iterations=iterations,
    )

    # Connect and load data
    logger.info(f"Connecting to {engine.name}...")
    engine.connect()
    summary.engine_version = engine.get_version()
    logger.info(f"Engine version: {summary.engine_version}")

    logger.info(f"Loading data from {data_dir}...")
    engine.load_data(data_dir, scale_factor)

    # Run warmup if requested
    if warmup:
        logger.info("Running warmup...")
        engine.warmup()

    # Get queries based on engine type
    if engine.uses_sql:
        # SQL-based engine: get queries for the engine's dialect
        all_queries = get_queries(engine.dialect)
    else:
        # Non-SQL engine: use query names q1-q12 with empty SQL
        all_queries = {f"q{i}": "" for i in range(1, 13)}

    # Filter queries if specific ones requested
    if queries:
        query_dict = {q: all_queries[q] for q in queries if q in all_queries}
        missing = set(queries) - set(query_dict.keys())
        if missing:
            logger.warning(f"Queries not found: {', '.join(missing)}")
    else:
        query_dict = all_queries

    query_names = sorted(query_dict.keys(), key=lambda x: int(x[1:]))
    total_runs = len(query_names) * iterations

    logger.info(
        f"Running {len(query_names)} queries × {iterations} iterations = {total_runs} total executions"
    )
    if not engine.uses_sql:
        logger.info(f"Engine '{engine.name}' uses function-based queries (not SQL)")
    print()  # Blank line before results

    # Run queries
    run_count = 0
    for iteration in range(1, iterations + 1):
        if iterations > 1:
            logger.info(f"=== Iteration {iteration}/{iterations} ===")

        for query_name in query_names:
            run_count += 1
            query_sql = query_dict[query_name]

            logger.debug(f"Running {query_name} ({run_count}/{total_runs})")
            result = engine.run_query(query_name, query_sql)
            result.iteration = iteration
            summary.add_result(result)

            # Print progress for each query
            status = "✓" if result.success else "✗"
            if result.success:
                print(
                    f"  [{status}] {query_name}: {result.duration_seconds:.4f}s ({result.row_count} rows)"
                )
            else:
                print(f"  [{status}] {query_name}: FAILED - {result.error_message}")

    return summary


def main() -> int:
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="SpatialBench Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --engine duckdb --data-dir ./data-sf1
  %(prog)s --engine duckdb --data-dir ./data-sf1 --iterations 5
  %(prog)s --engine all --data-dir ./data-sf1 --output results.json
  %(prog)s --list-engines

For more information, see: https://github.com/apache/sedona-spatialbench
        """,
    )

    # Engine selection
    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        help=f"Database engine to use. Use 'all' to run all engines. "
        f"Available: {', '.join(list_engines())}",
    )

    # Data configuration
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        help="Directory containing generated Parquet files",
    )

    parser.add_argument(
        "--scale-factor",
        "-s",
        type=float,
        default=1.0,
        help="Scale factor of the benchmark data (default: 1.0)",
    )

    # Query configuration
    parser.add_argument(
        "--queries",
        "-q",
        type=str,
        help="Comma-separated list of queries to run (e.g., q1,q2,q3). Default: all",
    )

    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=3,
        help="Number of iterations per query (default: 3)",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path for results (JSON or CSV based on extension)",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Output format when extension is ambiguous (default: json)",
    )

    # Execution options
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup before running benchmarks",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    # Information commands
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available database engines and exit",
    )

    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="List available benchmark queries and exit",
    )

    args = parser.parse_args()

    # Handle verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle information commands
    if args.list_engines:
        print("Available database engines:")
        for name in list_engines():
            engine = get_engine(name)
            query_type = "SQL" if engine.uses_sql else "function-based"
            print(f"  - {name} (dialect: {engine.dialect}, {query_type})")
        return 0

    if args.list_queries:
        print(f"Available benchmark queries ({get_query_count()} total):")
        # Show queries from SedonaSpark as reference
        for query_name in list_queries("SedonaSpark"):
            print(f"  - {query_name}")
        return 0

    # Validate required arguments
    if not args.engine:
        parser.error("--engine is required")
    if not args.data_dir:
        parser.error("--data-dir is required")

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1

    # Parse query list
    query_list = None
    if args.queries:
        query_list = [q.strip() for q in args.queries.split(",")]

    # Determine engines to run
    if args.engine.lower() == "all":
        engine_names = list_engines()
        logger.info(f"Running benchmarks on all engines: {', '.join(engine_names)}")
    else:
        engine_names = [args.engine.lower()]

    # Run benchmarks
    summaries: list[BenchmarkSummary] = []

    for engine_name in engine_names:
        try:
            engine = get_engine(engine_name)
        except ValueError as e:
            logger.error(str(e))
            return 1

        print(f"\n{'='*60}")
        print(f"Engine: {engine_name.upper()}")
        print(f"{'='*60}")

        try:
            with engine:
                summary = run_benchmark(
                    engine=engine,
                    data_dir=args.data_dir,
                    scale_factor=args.scale_factor,
                    iterations=args.iterations,
                    queries=query_list,
                    warmup=not args.no_warmup,
                )
                summaries.append(summary)
        except Exception as e:
            logger.error(f"Benchmark failed for {engine_name}: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            continue

    if not summaries:
        logger.error("No benchmarks completed successfully")
        return 1

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")

    for summary in summaries:
        print(f"Engine: {summary.engine} (v{summary.engine_version})")
        print(f"Scale Factor: {summary.scale_factor}")
        print(f"Iterations: {summary.iterations}")
        print(f"Successful: {summary.successful_queries}/{summary.total_queries}")
        print(f"Total Time: {summary.total_time:.2f}s")
        print()

        if summary.iterations > 1:
            print_aggregated_table(summary)
        else:
            print_results_table(summary.results)
        print()

    # Print comparison if multiple engines
    if len(summaries) > 1:
        print_comparison_table(summaries)

    # Save results if output specified
    if args.output:
        output_path = args.output

        # Determine format from extension or argument
        if output_path.suffix.lower() == ".csv":
            output_format = "csv"
        elif output_path.suffix.lower() == ".json":
            output_format = "json"
        else:
            output_format = args.output_format

        # For comparison mode, save all summaries
        if len(summaries) > 1:
            combined = {
                "comparison": True,
                "timestamp": datetime.now().isoformat(),
                "scale_factor": args.scale_factor,
                "iterations": args.iterations,
                "engines": [s.to_dict() for s in summaries],
            }
            if output_format == "json":
                import json

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(combined, f, indent=2)
            else:
                # For CSV, save aggregated results for all engines
                import csv

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "engine",
                        "query_name",
                        "mean_seconds",
                        "median_seconds",
                        "min_seconds",
                        "max_seconds",
                        "std_dev",
                        "success_rate",
                    ])
                    for summary in summaries:
                        agg = summary.get_aggregated_results()
                        for qname in sorted(agg.keys(), key=lambda x: int(x[1:])):
                            stats = agg[qname]
                            writer.writerow([
                                summary.engine,
                                qname,
                                stats["mean"],
                                stats["median"],
                                stats["min"],
                                stats["max"],
                                stats["std_dev"],
                                stats["success_rate"],
                            ])
        else:
            # Single engine output
            if output_format == "json":
                save_json(summaries[0], output_path)
            else:
                save_csv(summaries[0], output_path)

        logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
