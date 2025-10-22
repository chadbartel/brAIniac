"""
Test script to verify web search tools work correctly.
"""

# Standard Library
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# My Modules
from tools import search_web, search_wikipedia, get_current_date

print("=" * 60)
print("Testing Web Search Tools")
print("=" * 60)

# Test 1: Get current date
print("\n1. Testing get_current_date():")
print("-" * 40)
result = get_current_date()
print(result)

# Test 2: Search web
print("\n2. Testing search_web('George Santos'):")
print("-" * 40)
result = search_web("George Santos")
print(result)

# Test 3: Search Wikipedia
print("\n3. Testing search_wikipedia('George Santos'):")
print("-" * 40)
result = search_wikipedia("George Santos")
print(result[:500] + "..." if len(result) > 500 else result)

print("\n" + "=" * 60)
print("✓ All tool tests completed!")
print("=" * 60)
