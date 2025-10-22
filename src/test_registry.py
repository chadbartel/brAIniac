"""
Quick test to verify agent_registry module loads and works correctly.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from agent_registry import AgentRegistry, create_dynamic_selector

print("✓ Successfully imported agent_registry module")

# Test basic registry creation
config_list = [
    {
        "model": "test-model",
        "base_url": "http://localhost:8080/v1",
        "api_key": "not-needed",
        "price": [0.0, 0.0],
    }
]

llm_config = {"config_list": config_list, "temperature": 0.7}

registry = AgentRegistry(llm_config)
print("✓ Created AgentRegistry instance")

# Test agent registration
researcher = registry.register_agent(
    name="Researcher",
    system_message="You are a researcher.",
    capabilities=["research", "analysis"],
    can_route_to=["Writer"],
)
print(f"✓ Registered agent: {researcher.name}")

writer = registry.register_agent(
    name="Writer",
    system_message="You are a writer.",
    capabilities=["writing"],
)
print(f"✓ Registered agent: {writer.name}")

# Test retrieval
retrieved = registry.get_agent("Researcher")
assert retrieved is not None
assert retrieved.name == "Researcher"
print(f"✓ Retrieved agent: {retrieved.name}")

# Test capability search
researchers = registry.find_agents_by_capability("research")
assert len(researchers) == 1
assert researchers[0].name == "Researcher"
print(f"✓ Found {len(researchers)} agent(s) with 'research' capability")

# Test routing options
routes = registry.get_routing_options("Researcher")
assert len(routes) == 1
assert routes[0].name == "Writer"
print(f"✓ Researcher can route to: {[a.name for a in routes]}")

# Test list all agents
all_agents = registry.list_agents()
assert len(all_agents) == 2
print(f"✓ Total agents registered: {len(all_agents)}")

# Test speaker selector creation
selector = create_dynamic_selector(
    registry=registry,
    default_flow=["Researcher", "Writer"],
)
assert callable(selector)
print("✓ Created dynamic speaker selector")

print("\n" + "=" * 50)
print("All tests passed! ✅")
print("=" * 50)
