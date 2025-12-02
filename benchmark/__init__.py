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

A modular benchmark framework for executing SpatialBench queries against
various database engines and measuring performance.

To add support for a new database engine:
1. Create a new file in benchmark/engines/ (e.g., postgis_engine.py)
2. Subclass BenchmarkEngine from benchmark.engine_base
3. Implement all abstract methods: connect(), load_data(), run_query(), close()
4. Register your engine in benchmark/engines/__init__.py
"""

from benchmark.engine_base import BenchmarkEngine, QueryResult
from benchmark.engines import ENGINES, get_engine

__all__ = ["BenchmarkEngine", "QueryResult", "ENGINES", "get_engine"]
