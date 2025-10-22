# Quick Start Guide - Agent Registry POC

## Prerequisites

1. Docker running with llama.cpp server:

   ```bash
   docker-compose up -d
   ```

2. Python environment with autogen installed:

   ```bash
   poetry install
   # or
   pip install -r requirements.txt
   ```

## Running the POC

### Option 1: Run Main Script

```bash
python src/main.py
```

This will:

1. Wait for the model server to be ready
2. Create a registry with Researcher and Science_Expert agents
3. Run a conversation about photosynthesis
4. Save conversation logs to `logs/` directory

### Option 2: Interactive Python Session

```python
from agent_registry import AgentRegistry, create_dynamic_selector
import autogen

# Setup
config_list = [{
    "model": "local-model",
    "base_url": "http://localhost:8080/v1",
    "api_key": "not-needed",
    "price": [0.0, 0.0],
}]

llm_config = {"config_list": config_list, "temperature": 0.7}

# Create registry
registry = AgentRegistry(llm_config)

# Register your agents
coder = registry.register_agent(
    name="Coder",
    system_message="You write Python code.",
    capabilities=["coding", "debugging"],
    can_route_to=["Tester"],
)

tester = registry.register_agent(
    name="Tester",
    system_message="You test code.",
    capabilities=["testing", "validation"],
    can_route_to=["Coder"],
)

# Create conversation
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
)

selector = create_dynamic_selector(
    registry=registry,
    default_flow=["User", "Coder", "Tester"],
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, tester],
    messages=[],
    max_round=10,
    speaker_selection_method=selector,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Start conversation
user_proxy.initiate_chat(
    manager,
    message="Write a function to calculate fibonacci numbers and test it."
)
```

## Understanding the Output

### Console Output

```
2025-10-22 06:27:36 - root - INFO - Polling server at http://localhost:8080/v1...
2025-10-22 06:27:41 - root - INFO - ✓ Model server ready!
2025-10-22 06:27:41 - root - INFO - Registered agent 'Researcher' with capabilities: ['research', 'data_gathering', 'analysis']
2025-10-22 06:27:41 - root - INFO - Registered agent 'Science_Expert' with capabilities: ['explanation', 'simplification', 'teaching']
2025-10-22 06:27:41 - root - INFO - Starting conversation: Research photosynthesis

User_Proxy (to chat_manager):
Research the process of photosynthesis...

Researcher (to chat_manager):
Photosynthesis is the process...
NEXT: Science_Expert

Science_Expert (to chat_manager):
In simple terms, photosynthesis is...
TERMINATE

2025-10-22 06:28:30 - root - INFO - 📝 Turn 1: User_Proxy
2025-10-22 06:28:30 - root - INFO - 📝 Turn 2: Researcher
2025-10-22 06:28:30 - root - INFO - 📝 Turn 3: Science_Expert
2025-10-22 06:28:30 - root - INFO - 💾 Full conversation saved to: logs/full_conversation_20251022_062830.json
2025-10-22 06:28:30 - root - INFO - ✓ Conversation complete
```

### Log Files

Two types of logs are created in `logs/`:

1. **JSONL format** (`conversation_YYYYMMDD_HHMMSS.jsonl`):

   ```jsonl
   {"turn": 1, "timestamp": "2025-10-22T06:28:30", "agent": "User_Proxy", "message": "Research..."}
   {"turn": 2, "timestamp": "2025-10-22T06:28:45", "agent": "Researcher", "message": "Photosynthesis..."}
   ```

2. **JSON format** (`full_conversation_YYYYMMDD_HHMMSS.json`):

   ```json
   {
     "timestamp": "2025-10-22T06:28:30",
     "messages": [...],
     "total_turns": 3
   }
   ```

## Key Features to Try

### 1. Explicit Routing

Agents use "NEXT: AgentName" to explicitly route:

```python
system_message = """
After your research, end with: 'NEXT: Science_Expert'
"""
```

### 2. Capability-Based Routing

Messages with keywords automatically route to capable agents:

```python
# "I need to research..." → Routes to agent with "research" capability
# "Please explain..." → Routes to agent with "explanation" capability
```

### 3. Dynamic Agent Addition

Add agents at runtime:

```python
# Check if specialized agent is needed
if "security" in user_message.lower():
    registry.register_agent(
        name="SecurityExpert",
        system_message="You audit code for security issues.",
        capabilities=["security", "auditing"],
    )
```

### 4. Query the Registry

Inspect registered agents:

```python
# Get all agents
all_agents = registry.list_agents()

# Find agents by capability
researchers = registry.find_agents_by_capability("research")

# Check routing options
next_agents = registry.get_routing_options("Researcher")

# Get agent capabilities
caps = registry.get_agent_capabilities("Researcher")
```

## Common Workflows

### Research → Analysis → Report

```python
registry.register_agent(
    name="Researcher",
    capabilities=["research"],
    can_route_to=["Analyst"],
)

registry.register_agent(
    name="Analyst",
    capabilities=["analysis"],
    can_route_to=["Reporter"],
)

registry.register_agent(
    name="Reporter",
    capabilities=["reporting"],
)

selector = create_dynamic_selector(
    registry=registry,
    default_flow=["User", "Researcher", "Analyst", "Reporter"],
)
```

### Code → Review → Test

```python
registry.register_agent(
    name="Developer",
    capabilities=["coding"],
    can_route_to=["Reviewer"],
)

registry.register_agent(
    name="Reviewer",
    capabilities=["code_review"],
    can_route_to=["Tester"],
)

registry.register_agent(
    name="Tester",
    capabilities=["testing"],
)
```

## Troubleshooting

### Server Not Ready

```
✗ Failed to connect after 30 attempts
```

**Solution:** Ensure Docker container is running:

```bash
docker-compose ps
docker-compose up -d
```

### Agent Not Found

```
WARNING - Agent 'AgentName' not found in registry
```

**Solution:** Check agent is registered:

```python
agent = registry.get_agent("AgentName")
if agent is None:
    print("Agent not registered!")
```

### Wrong Agent Selected

Enable DEBUG logging to see selection logic:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Output will show why each agent was selected:

```
DEBUG - Selecting next speaker after 'Researcher'
INFO - Following routing rule: 'Researcher' -> 'Science_Expert'
```

## Next Steps

1. **Read the Full Documentation**: See `src/AGENT_REGISTRY.md`
2. **Study Examples**: Run `src/examples_agent_registry.py`
3. **Customize for Your Use Case**: Modify agent system messages and capabilities
4. **Add More Agents**: Register specialized agents for your domain
5. **Implement Advanced Features**: Agent pools, priorities, state-based routing

## Key Files

- `src/agent_registry.py` - Core registry implementation
- `src/main.py` - Working POC example
- `src/AGENT_REGISTRY.md` - Complete API documentation
- `src/examples_agent_registry.py` - Usage examples
- `IMPLEMENTATION_SUMMARY.md` - High-level overview
