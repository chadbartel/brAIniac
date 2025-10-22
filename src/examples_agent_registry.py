"""
Example demonstrating advanced agent registry features.

This script shows how to:
1. Dynamically register agents at runtime
2. Use capability-based routing
3. Add new agents on-demand
4. Query the registry for agent information
"""

import logging
import autogen
from agent_registry import AgentRegistry, create_dynamic_selector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_1_basic_registry():
    """Example 1: Basic agent registration and retrieval."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Agent Registry")
    logger.info("=" * 60)

    config_list = [
        {
            "model": "local-model",
            "base_url": "http://localhost:8080/v1",
            "api_key": "not-needed",
            "price": [0.0, 0.0],
        }
    ]

    llm_config = {"config_list": config_list, "temperature": 0.7}

    # Create registry
    registry = AgentRegistry(llm_config)

    # Register multiple agents
    researcher = registry.register_agent(
        name="Researcher",
        system_message="You are a research assistant.",
        capabilities=["research", "analysis"],
    )

    writer = registry.register_agent(
        name="Writer",
        system_message="You are a technical writer.",
        capabilities=["writing", "documentation"],
    )

    # Retrieve agents
    retrieved = registry.get_agent("Researcher")
    logger.info("Retrieved agent: %s", retrieved.name)

    # List all agents
    all_agents = registry.list_agents()
    logger.info("Total agents registered: %d", len(all_agents))


def example_2_capability_based_routing():
    """Example 2: Finding agents by capability."""
    logger.info("=" * 60)
    logger.info("Example 2: Capability-Based Routing")
    logger.info("=" * 60)

    config_list = [
        {
            "model": "local-model",
            "base_url": "http://localhost:8080/v1",
            "api_key": "not-needed",
            "price": [0.0, 0.0],
        }
    ]

    llm_config = {"config_list": config_list, "temperature": 0.7}

    registry = AgentRegistry(llm_config)

    # Register agents with different capabilities
    registry.register_agent(
        name="DataScientist",
        system_message="You analyze data and build models.",
        capabilities=["analysis", "modeling", "statistics"],
    )

    registry.register_agent(
        name="Researcher",
        system_message="You research topics thoroughly.",
        capabilities=["research", "analysis", "fact_checking"],
    )

    registry.register_agent(
        name="Visualizer",
        system_message="You create data visualizations.",
        capabilities=["visualization", "charting"],
    )

    # Find all agents with 'analysis' capability
    analysts = registry.find_agents_by_capability("analysis")
    logger.info(
        "Agents with 'analysis' capability: %s",
        [agent.name for agent in analysts],
    )

    # Find agents with 'visualization' capability
    visualizers = registry.find_agents_by_capability("visualization")
    logger.info(
        "Agents with 'visualization' capability: %s",
        [agent.name for agent in visualizers],
    )


def example_3_routing_rules():
    """Example 3: Using routing rules for agent transitions."""
    logger.info("=" * 60)
    logger.info("Example 3: Routing Rules")
    logger.info("=" * 60)

    config_list = [
        {
            "model": "local-model",
            "base_url": "http://localhost:8080/v1",
            "api_key": "not-needed",
            "price": [0.0, 0.0],
        }
    ]

    llm_config = {"config_list": config_list, "temperature": 0.7}

    registry = AgentRegistry(llm_config)

    # Register agents with routing rules
    registry.register_agent(
        name="Planner",
        system_message="You create project plans.",
        capabilities=["planning"],
        can_route_to=["Developer", "Tester"],
    )

    registry.register_agent(
        name="Developer",
        system_message="You write code.",
        capabilities=["coding"],
        can_route_to=["Tester"],
    )

    registry.register_agent(
        name="Tester",
        system_message="You test software.",
        capabilities=["testing"],
        can_route_to=["Planner"],
    )

    # Check routing options
    planner_routes = registry.get_routing_options("Planner")
    logger.info(
        "Planner can route to: %s",
        [agent.name for agent in planner_routes],
    )

    developer_routes = registry.get_routing_options("Developer")
    logger.info(
        "Developer can route to: %s",
        [agent.name for agent in developer_routes],
    )


def example_4_dynamic_agent_creation():
    """Example 4: Adding agents dynamically during runtime."""
    logger.info("=" * 60)
    logger.info("Example 4: Dynamic Agent Creation")
    logger.info("=" * 60)

    config_list = [
        {
            "model": "local-model",
            "base_url": "http://localhost:8080/v1",
            "api_key": "not-needed",
            "price": [0.0, 0.0],
        }
    ]

    llm_config = {"config_list": config_list, "temperature": 0.7}

    registry = AgentRegistry(llm_config)

    # Start with base agents
    registry.register_agent(
        name="Coordinator",
        system_message="You coordinate tasks.",
        capabilities=["coordination"],
    )

    logger.info("Initial agents: %d", len(registry.list_agents()))

    # Dynamically add a specialist based on need
    if "analysis" in ["analysis", "research"]:  # Simulated condition
        registry.register_agent(
            name="Analyst",
            system_message="You analyze complex data.",
            capabilities=["analysis", "statistics"],
        )
        logger.info("Added Analyst agent dynamically")

    logger.info("After dynamic creation: %d", len(registry.list_agents()))

    # Add another specialist
    registry.register_agent(
        name="SecurityExpert",
        system_message="You review security concerns.",
        capabilities=["security", "auditing"],
    )

    logger.info("Final agent count: %d", len(registry.list_agents()))


def example_5_speaker_selector():
    """Example 5: Using the dynamic speaker selector."""
    logger.info("=" * 60)
    logger.info("Example 5: Dynamic Speaker Selector")
    logger.info("=" * 60)

    config_list = [
        {
            "model": "local-model",
            "base_url": "http://localhost:8080/v1",
            "api_key": "not-needed",
            "price": [0.0, 0.0],
        }
    ]

    llm_config = {"config_list": config_list, "temperature": 0.7}

    registry = AgentRegistry(llm_config)

    # Register agents
    registry.register_agent(
        name="Collector",
        system_message="You collect information.",
        capabilities=["research"],
        can_route_to=["Analyzer"],
    )

    registry.register_agent(
        name="Analyzer",
        system_message="You analyze information.",
        capabilities=["analysis"],
        can_route_to=["Reporter"],
    )

    registry.register_agent(
        name="Reporter",
        system_message="You report findings.",
        capabilities=["reporting"],
    )

    # Create selector with default flow
    selector = create_dynamic_selector(
        registry=registry,
        default_flow=["Collector", "Analyzer", "Reporter"],
    )

    logger.info(
        "Created speaker selector with default flow: %s",
        ["Collector", "Analyzer", "Reporter"],
    )

    # In a real scenario, this would be used in a GroupChat
    # groupchat = autogen.GroupChat(
    #     agents=registry.list_agents(),
    #     messages=[],
    #     max_round=10,
    #     speaker_selection_method=selector,
    # )


if __name__ == "__main__":
    logger.info("Agent Registry Examples")
    logger.info("=" * 60)

    # Run examples
    example_1_basic_registry()
    print()

    example_2_capability_based_routing()
    print()

    example_3_routing_rules()
    print()

    example_4_dynamic_agent_creation()
    print()

    example_5_speaker_selector()
    print()

    logger.info("=" * 60)
    logger.info("All examples completed successfully!")
