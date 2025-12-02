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
Tests for the query provider module.

These tests verify:
1. Queries are loaded correctly from print_queries.py
2. All 12 benchmark queries (q1-q12) are available
3. Different dialects return appropriate SQL variations
4. Query listing functions work correctly
"""

import pytest

from benchmark.query_provider import (
    get_queries,
    list_queries,
    get_query,
    get_query_count,
)


class TestGetQueries:
    """Tests for the get_queries() function."""

    def test_returns_dict_of_queries(self):
        """get_queries() returns a dictionary of query name -> SQL."""
        queries = get_queries("DuckDB")

        assert isinstance(queries, dict)
        assert all(isinstance(k, str) for k in queries.keys())
        assert all(isinstance(v, str) for v in queries.values())

    def test_returns_all_12_queries(self):
        """get_queries() returns all 12 benchmark queries."""
        queries = get_queries("DuckDB")

        expected = {f"q{i}" for i in range(1, 13)}
        assert set(queries.keys()) == expected

    def test_queries_contain_sql(self):
        """Each query contains valid SQL keywords."""
        queries = get_queries("DuckDB")

        for name, sql in queries.items():
            assert sql.strip(), f"Query {name} is empty"
            # All queries should have SELECT
            assert "SELECT" in sql.upper(), f"Query {name} missing SELECT"

    @pytest.mark.parametrize("dialect", ["DuckDB", "SedonaSpark", "Databricks"])
    def test_supports_multiple_dialects(self, dialect):
        """get_queries() supports multiple SQL dialects."""
        queries = get_queries(dialect)

        assert len(queries) == 12

    def test_unknown_dialect_raises_error(self):
        """get_queries() raises error for unknown dialect."""
        with pytest.raises(ValueError, match="Unknown dialect"):
            get_queries("NonexistentDialect")

    def test_dialect_case_sensitivity(self):
        """Dialect names are case-sensitive (must match class names)."""
        # These should work (exact class names)
        get_queries("DuckDB")
        get_queries("SedonaSpark")

        # These should fail (wrong case)
        with pytest.raises(ValueError):
            get_queries("duckdb")


class TestListQueries:
    """Tests for the list_queries() function."""

    def test_returns_sorted_query_names(self):
        """list_queries() returns query names in sorted order."""
        queries = list_queries("DuckDB")

        # Should be sorted by query number
        expected = [f"q{i}" for i in range(1, 13)]
        assert queries == expected

    def test_returns_list(self):
        """list_queries() returns a list type."""
        queries = list_queries("DuckDB")

        assert isinstance(queries, list)


class TestGetQuery:
    """Tests for the get_query() function."""

    def test_returns_single_query_sql(self):
        """get_query() returns SQL for a specific query."""
        sql = get_query("DuckDB", "q1")

        assert isinstance(sql, str)
        assert "SELECT" in sql.upper()

    def test_all_queries_retrievable(self):
        """All 12 queries can be retrieved individually."""
        for i in range(1, 13):
            sql = get_query("DuckDB", f"q{i}")
            assert sql is not None
            assert len(sql) > 0

    def test_unknown_query_returns_none(self):
        """get_query() returns None for unknown query name."""
        result = get_query("DuckDB", "q99")

        assert result is None


class TestGetQueryCount:
    """Tests for the get_query_count() function."""

    def test_returns_12(self):
        """get_query_count() returns 12 (the number of benchmark queries)."""
        count = get_query_count()

        assert count == 12


class TestQueryDialectDifferences:
    """Tests verifying that different dialects produce different SQL where expected."""

    def test_duckdb_uses_st_functions(self):
        """DuckDB dialect uses ST_ prefixed spatial functions."""
        queries = get_queries("DuckDB")

        # At least some queries should use ST_ functions
        spatial_queries = [q for q in queries.values() if "ST_" in q]
        assert len(spatial_queries) > 0, "DuckDB should use ST_ spatial functions"

    def test_sedonaspark_dialect_available(self):
        """SedonaSpark dialect is available and returns queries."""
        queries = get_queries("SedonaSpark")

        assert len(queries) == 12

    def test_databricks_dialect_available(self):
        """Databricks dialect is available and returns queries."""
        queries = get_queries("Databricks")

        assert len(queries) == 12


class TestQueryConsistency:
    """Tests for query consistency across dialects."""

    def test_all_dialects_have_same_query_names(self):
        """All dialects have the same query names (q1-q12)."""
        duckdb_queries = set(get_queries("DuckDB").keys())
        sedona_queries = set(get_queries("SedonaSpark").keys())
        databricks_queries = set(get_queries("Databricks").keys())

        assert duckdb_queries == sedona_queries == databricks_queries

    def test_query_names_are_lowercase(self):
        """All query names follow q{n} format (lowercase)."""
        for dialect in ["DuckDB", "SedonaSpark", "Databricks"]:
            queries = get_queries(dialect)
            for name in queries.keys():
                assert name.startswith("q"), f"Query name should start with 'q': {name}"
                assert name[1:].isdigit(), f"Query name should be q followed by number: {name}"
