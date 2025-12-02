# SpatialBench Copilot Instructions

## Project Overview

SpatialBench is a Rust-based benchmark for evaluating geospatial SQL analytics across database systems. It's a fork of tpchgen-rs with added spatial data generation using the Spider algorithm.

### Component Architecture

```
spatialbench/         # Core Rust library (zero dependencies by design)
├── generators.rs     # Table generators (Vehicle, Driver, Customer, Trip, Building)
├── spatial/          # Spider geometry generation (Point, Box, Polygon)
│   ├── generator.rs  # SpatialGenerator dispatches to distribution types
│   ├── distributions.rs  # Uniform, Normal, Diagonal, Bit, Sierpinski, Thomas, HierThomas
│   └── config.rs     # SpatialConfig for YAML parsing

spatialbench-arrow/   # Arrow RecordBatch generation (minimal deps: arrow crate only)
├── trip.rs, building.rs, etc.  # Arrow schema definitions per table

spatialbench-cli/     # Rust CLI for data generation (clap, parquet, tokio)
├── main.rs           # CLI entry point, config resolution, table generation
├── zone/             # Zone table generation (special handling, async)
└── spatial_config_file.rs  # YAML config parsing

benchmark/            # Python benchmark runner framework
├── engine_base.py    # Abstract BenchmarkEngine base class
├── query_provider.py # Loads queries from print_queries.py
├── reporting.py      # Result formatting, JSON/CSV export
└── engines/          # Database engine implementations
    ├── duckdb_engine.py      # DuckDB (reference SQL implementation)
    ├── sedona_engine.py      # Apache Sedona / Spark
    ├── databricks_engine.py  # Databricks SQL warehouse
    ├── geopandas_engine.py   # GeoPandas (function-based)
    └── polars_engine.py      # Polars + spatial_polars (function-based)

print_queries.py      # SQL query definitions for all dialects
geopandas.py          # GeoPandas query functions (q1-q12)
spatial_polars.py     # Polars query functions (q1-q12)
runner.py             # CLI entry point for benchmark execution
```

## Key Patterns

### Generator Pattern
All table generators follow the same pattern with deterministic seeding:
```rust
let generator = TripGenerator::new(scale_factor, part, num_parts);
for trip in generator.iter().take(100) {
    // Process trip
}
```

### Spatial Generation Flow
1. `SpatialGenerator::generate(index, continent_affine)` → dispatches to distribution
2. Distribution generates point in unit square `[0,1]²`
3. Affine transform maps to continent bounding box
4. `geom_type` (point/box/polygon) determines final geometry

### Configuration Resolution
1. `--config <path>` explicit YAML
2. `./spatialbench-config.yml` local default
3. Built-in defaults from `spatial/defaults.rs`

## Development Commands

```bash
# Run all Rust tests
cargo test

# Run tests for specific crate
cargo test -p spatialbench --tests
cargo test -p spatialbench --doc
cargo test -p spatialbench-arrow
cargo test -p spatialbench-cli

# Lint (required before PR)
cargo fmt --all -- --check
cargo clippy -- -D warnings

# Run CLI locally (data generation)
cargo run --bin spatialbench-cli -- --scale-factor 1 --tables trip

# Debug with verbose logging
cargo run --bin spatialbench-cli -- --scale-factor 1 --verbose
RUST_LOG=debug cargo run --bin spatialbench-cli -- --scale-factor 1

# Run benchmark queries (Python)
pip install -r benchmark-requirements.txt
python runner.py --engine duckdb --data-dir ./data-sf1 --iterations 3

# Print SQL queries for a specific dialect
python print_queries.py DuckDB
```

## Performance Constraints

The `spatialbench` crate has **no external dependencies by design** for embeddability. Avoid:
- Heap allocations during iteration (use stack buffers)
- Floating-point for display (use integer arithmetic)
- Adding dependencies to core crate

Use `spatialbench-arrow` or `spatialbench-cli` for features requiring external crates.

## Table Schema

| Table    | Scale Factor Multiplier | Spatial Column          |
|----------|------------------------|-------------------------|
| Trip     | 6,000,000 × SF         | pickup/dropoff Points   |
| Building | 20,000 × (1 + log₂SF)  | boundary Polygon        |
| Zone     | Tiered by SF range     | boundary Polygon        |
| Customer | 30,000 × SF            | None                    |
| Driver   | 500 × SF               | None                    |
| Vehicle  | 100 × SF               | None                    |

## Testing Expectations

- Doc tests in `lib.rs` show expected TBL output format
- Integration tests verify `IntoIterator` implementations
- Deterministic output: same seed + index = same geometry

## Adding New Distribution Types

1. Add variant to `DistributionType` enum in `spatial/config.rs`
2. Implement `generate_<type>()` in `spatial/distributions.rs`
3. Add match arm in `SpatialGenerator::generate()`
4. Add params variant in `SpatialParams` enum
5. Update `CONFIGURATION.md` with parameter documentation

## Adding New Benchmark Engines

To support a new database (e.g., PostGIS, BigQuery):

1. Create `benchmark/engines/<name>_engine.py`
2. Subclass `BenchmarkEngine` from `benchmark.engine_base`
3. Implement required methods: `connect()`, `load_data()`, `run_query()`, `close()`
4. Set `dialect` property to match a dialect in `print_queries.py`
5. Set `uses_sql` property to `False` for function-based engines
6. Register in `benchmark/engines/__init__.py`

### SQL-based Engine Example
```python
from benchmark.engine_base import BenchmarkEngine, QueryResult

class MyEngine(BenchmarkEngine):
    @property
    def name(self) -> str:
        return "myengine"

    @property
    def dialect(self) -> str:
        return "SedonaSpark"  # Must match print_queries.py

    def connect(self) -> None: ...
    def load_data(self, data_dir, scale_factor) -> None: ...
    def run_query(self, query_name, query_sql) -> QueryResult: ...
    def close(self) -> None: ...
```

### Function-based Engine Example
For engines that don't use SQL (like GeoPandas or Polars):
```python
class MyEngine(BenchmarkEngine):
    @property
    def uses_sql(self) -> bool:
        return False  # Query functions instead of SQL

    def run_query(self, query_name, query_sql) -> QueryResult:
        # query_sql is ignored; call function by query_name
        func = self._query_funcs[query_name]
        result_df = func(self._data_paths)
        return QueryResult(...)
```

### Available Engines

| Engine     | Dialect      | Type           | Dependencies                     |
|------------|--------------|----------------|----------------------------------|
| duckdb     | DuckDB       | SQL            | duckdb                           |
| sedona     | SedonaSpark  | SQL            | pyspark, apache-sedona           |
| sedonadb   | SedonaDB     | SQL            | apache-sedona, sedonadb          |
| databricks | Databricks   | SQL            | databricks-sql-connector         |
| geopandas  | Geopandas    | function-based | geopandas, pandas, shapely       |
| polars     | Polars       | function-based | polars, spatial-polars           |

## SQL Dialect System

Queries are defined in `print_queries.py` using inheritance:
- `SpatialBenchBenchmark` - Base class (SedonaSpark dialect)
- `DuckDBSpatialBenchBenchmark` - Overrides for DuckDB-specific syntax
- `DatabricksSpatialBenchBenchmark` - Overrides for Databricks
- `SedonaDBSpatialBenchBenchmark` - Overrides for SedonaDB

Only override methods for queries that differ from the base dialect.
