# Agent Registry Pattern

The agent registry provides a dynamic, scalable approach to managing autogen agents with capability-based routing and on-demand agent creation.

## Features

- **Dynamic Agent Registration**: Register agents at runtime with capabilities and routing rules
- **Capability-Based Routing**: Find agents by their capabilities (e.g., "research", "analysis")
- **Routing Rules**: Define which agents can route to which other agents
- **Speaker Selection**: Custom speaker selection logic that uses registry metadata
- **Scalable**: Add/remove agents without modifying core conversation logic

## Quick Start

### Basic Usage

```python
from agent_registry import AgentRegistry, create_dynamic_selector
import autogen

# Configure LLM
config_list = [{
    "model": "local-model",
    "base_url": "http://localhost:8080/v1",
    "api_key": "not-needed",
    "price": [0.0, 0.0],
}]

llm_config = {"config_list": config_list, "temperature": 0.7}

# Initialize registry
registry = AgentRegistry(llm_config)

# Register agents with capabilities
researcher = registry.register_agent(
    name="Researcher",
    system_message="You are a research assistant.",
    capabilities=["research", "data_gathering"],
    can_route_to=["Analyst"],
)

analyst = registry.register_agent(
    name="Analyst",
    system_message="You analyze data.",
    capabilities=["analysis", "statistics"],
    can_route_to=["Reporter"],
)

# Create speaker selector
selector = create_dynamic_selector(
    registry=registry,
    default_flow=["Researcher", "Analyst", "Reporter"],
)

# Use in GroupChat
groupchat = autogen.GroupChat(
    agents=registry.list_agents(),
    messages=[],
    max_round=10,
    speaker_selection_method=selector,
)
```

## API Reference

### AgentRegistry

#### `__init__(llm_config: Dict)`

Initialize the registry with LLM configuration.

#### `register_agent(name, system_message, capabilities=None, can_route_to=None)`

Register a new agent.

**Parameters:**

- `name` (str): Unique agent name
- `system_message` (str): System prompt for the agent
- `capabilities` (List[str], optional): List of capabilities (e.g., ["research", "coding"])
- `can_route_to` (List[str], optional): List of agent names this agent can route to

**Returns:** `autogen.AssistantAgent`

**Example:**

```python
researcher = registry.register_agent(
    name="Researcher",
    system_message="You research topics thoroughly.",
    capabilities=["research", "fact_checking"],
    can_route_to=["Writer", "Analyst"],
)
```

#### `get_agent(name: str) -> Optional[Agent]`

Retrieve an agent by name.

#### `find_agents_by_capability(capability: str) -> List[Agent]`

Find all agents with a specific capability.

**Example:**

```python
# Find all agents that can do research
researchers = registry.find_agents_by_capability("research")
```

#### `get_routing_options(agent_name: str) -> List[Agent]`

Get agents that a specific agent can route to.

#### `list_agents() -> List[Agent]`

Get all registered agents.

#### `get_agent_capabilities(agent_name: str) -> List[str]`

Get the capabilities of a specific agent.

### create_dynamic_selector

#### `create_dynamic_selector(registry, default_flow=None)`

Create a speaker selection function for GroupChat.

**Parameters:**

- `registry` (AgentRegistry): The agent registry
- `default_flow` (List[str], optional): Default flow of agent names

**Returns:** Callable speaker selection function

**Selection Logic Priority:**

1. Explicit routing in message (e.g., "NEXT: AgentName")
2. Routing rules from registry (`can_route_to`)
3. Capability-based keywords in message
4. Default flow if provided
5. Round-robin fallback

**Example:**

```python
selector = create_dynamic_selector(
    registry=registry,
    default_flow=["Planner", "Developer", "Tester"],
)

groupchat = autogen.GroupChat(
    agents=registry.list_agents(),
    speaker_selection_method=selector,
)
```

## Advanced Usage

### Capability-Based Routing

Agents are automatically selected based on keywords in messages:

```python
# Register agents with different capabilities
registry.register_agent(
    name="Researcher",
    system_message="...",
    capabilities=["research", "investigation"],
)

registry.register_agent(
    name="Explainer",
    system_message="...",
    capabilities=["explanation", "teaching"],
)

# Message with "research" keyword will route to Researcher
# Message with "explain" keyword will route to Explainer
```

### Explicit Routing

Agents can explicitly route to others using "NEXT:" syntax:

```python
system_message = """
You are a researcher. After gathering information, end your
message with: 'NEXT: Analyst' to pass to the analyst.
"""
```

### Dynamic Agent Addition

Add agents at runtime based on conversation needs:

```python
def add_specialist_if_needed(registry, topic):
    if topic == "security":
        registry.register_agent(
            name="SecurityExpert",
            system_message="You are a security specialist.",
            capabilities=["security", "auditing"],
        )
    elif topic == "performance":
        registry.register_agent(
            name="PerformanceExpert",
            system_message="You optimize performance.",
            capabilities=["optimization", "profiling"],
        )
```

## Examples

See `examples_agent_registry.py` for comprehensive examples:

```bash
python src/examples_agent_registry.py
```

## Architecture

```
┌─────────────────────────────────────────┐
│         AgentRegistry                    │
├─────────────────────────────────────────┤
│ - _agents: Dict[str, Agent]             │
│ - _capabilities: Dict[str, List[str]]   │
│ - _routing_rules: Dict[str, List[str]]  │
├─────────────────────────────────────────┤
│ + register_agent()                      │
│ + get_agent()                           │
│ + find_agents_by_capability()           │
│ + get_routing_options()                 │
└─────────────────────────────────────────┘
                  │
                  │ uses
                  ▼
┌─────────────────────────────────────────┐
│    create_dynamic_selector()            │
├─────────────────────────────────────────┤
│ Returns: speaker_selection_function     │
│                                         │
│ Selection Priority:                     │
│ 1. Explicit routing (NEXT: Agent)      │
│ 2. Routing rules (can_route_to)        │
│ 3. Capability keywords                  │
│ 4. Default flow                         │
│ 5. Round-robin fallback                 │
└─────────────────────────────────────────┘
                  │
                  │ used by
                  ▼
┌─────────────────────────────────────────┐
│        autogen.GroupChat                │
├─────────────────────────────────────────┤
│ speaker_selection_method=selector       │
└─────────────────────────────────────────┘
```

## Benefits Over Static Configuration

| Feature | Static Config | Registry Pattern |
|---------|--------------|------------------|
| Add agents at runtime | ❌ | ✅ |
| Capability-based routing | ❌ | ✅ |
| Query agent metadata | ❌ | ✅ |
| Dynamic flow changes | ❌ | ✅ |
| Multi-tenant support | ❌ | ✅ |
| Agent reusability | ❌ | ✅ |

## Best Practices

1. **Use Clear Capability Names**: Use consistent, descriptive capability names like "research", "analysis", "explanation"

2. **Define Routing Rules**: Specify `can_route_to` for predictable conversation flow

3. **Leverage Default Flow**: Provide a default flow as fallback for common scenarios

4. **Log Selection Decisions**: Enable INFO logging to see why agents are selected

5. **Keep System Messages Clear**: Include routing instructions in system messages

## Troubleshooting

### Agent Not Being Selected

- Check that agent is registered: `registry.get_agent("AgentName")`
- Verify capabilities: `registry.get_agent_capabilities("AgentName")`
- Enable DEBUG logging to see selection logic

### Unexpected Speaker Selection

- Review routing rules: `registry.get_routing_options("AgentName")`
- Check message content for keyword triggers
- Verify default flow order

### Agent Not Found Error

- Ensure agent is registered before use
- Check for typos in agent names (case-sensitive)
- Verify agent isn't accidentally removed

## Contributing

When adding new features to the registry:

1. Update `AgentRegistry` class with new methods
2. Add corresponding logic to `create_dynamic_selector`
3. Write examples in `examples_agent_registry.py`
4. Update this README with API documentation
5. Add tests for new functionality
