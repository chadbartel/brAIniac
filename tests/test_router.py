#!/usr/bin/env python3
"""Quick test to validate router prompt fix."""
import sys
sys.path.insert(0, "servers/system-server/src")

from system_server.orchestrator import _call_tool_router

# Test queries that should trigger web_search_needed=True
test_queries = [
    "what is the current weather in seattle",
    "what is today's weather in chicago",
    "latest news on AI",
    "current stock price of AAPL",
    "hello how are you",  # Should be False
    "what is 2+2",  # Should be False
]

print("Testing router responses:\n")
for query in test_queries:
    result = _call_tool_router(query, history=None)
    status = "✅" if result.web_search_needed else "❌"
    print(f"{status} '{query}'")
    print(f"   web_search_needed={result.web_search_needed}, query='{result.query}'")
    print()
