"""
Dynamic agent registry for runtime agent management.

This module provides a registry pattern for managing agents dynamically,
including capability-based routing and on-demand agent creation.
"""

# Standard Library
import logging
from typing import Dict, List, Optional, Callable

# Third Party
import autogen

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for dynamically managing agents in conversations."""

    def __init__(self, llm_config: Dict):
        """
        Initialize the agent registry.

        Args:
            llm_config: Configuration for LLM agents
        """
        self.llm_config = llm_config
        self._agents: Dict[str, autogen.Agent] = {}
        self._capabilities: Dict[str, List[str]] = {}
        self._routing_rules: Dict[str, List[str]] = {}

    def register_agent(
        self,
        name: str,
        system_message: str,
        capabilities: Optional[List[str]] = None,
        can_route_to: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None,
    ) -> autogen.AssistantAgent:
        """
        Register a new agent with the registry.

        Args:
            name: Unique agent name
            system_message: System message defining agent behavior
            capabilities: List of capabilities (e.g., ["research",
                "analysis"])
            can_route_to: List of agent names this agent can route to
            tools: List of tool definitions for function calling

        Returns:
            The created agent
        """
        if name in self._agents:
            logger.warning("Agent '%s' already registered, overwriting", name)

        # Create llm_config with tools if provided
        agent_llm_config = self.llm_config.copy()
        if tools:
            agent_llm_config["tools"] = tools

        agent = autogen.AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=agent_llm_config,
        )

        self._agents[name] = agent
        self._capabilities[name] = capabilities or []
        self._routing_rules[name] = can_route_to or []

        logger.info(
            "Registered agent '%s' with capabilities: %s",
            name,
            capabilities,
        )

        return agent

    def get_agent(self, name: str) -> Optional[autogen.Agent]:
        """
        Retrieve an agent by name.

        Args:
            name: Agent name

        Returns:
            The agent if found, None otherwise
        """
        return self._agents.get(name)

    def find_agents_by_capability(
        self,
        capability: str,
    ) -> List[autogen.Agent]:
        """
        Find all agents with a specific capability.

        Args:
            capability: The capability to search for

        Returns:
            List of agents with that capability
        """
        return [
            self._agents[name]
            for name, caps in self._capabilities.items()
            if capability in caps
        ]

    def get_routing_options(self, agent_name: str) -> List[autogen.Agent]:
        """
        Get the list of agents that a specific agent can route to.

        Args:
            agent_name: Name of the agent

        Returns:
            List of agents that can be routed to
        """
        routing_names = self._routing_rules.get(agent_name, [])
        return [
            self._agents[name]
            for name in routing_names
            if name in self._agents
        ]

    def list_agents(self) -> List[autogen.Agent]:
        """
        Get all registered agents.

        Returns:
            List of all agents
        """
        return list(self._agents.values())

    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """
        Get the capabilities of a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of capabilities
        """
        return self._capabilities.get(agent_name, [])


def create_dynamic_selector(
    registry: AgentRegistry,
    default_flow: Optional[List[str]] = None,
) -> Callable:
    """
    Create a speaker selection function that uses the registry.

    Args:
        registry: The agent registry
        default_flow: Optional default flow of agent names

    Returns:
        A speaker selection function
    """

    def select_speaker(
        last_speaker: autogen.Agent,
        groupchat: autogen.GroupChat,
    ) -> autogen.Agent:
        """
        Select next speaker dynamically based on registry.

        Args:
            last_speaker: The agent who just spoke
            groupchat: The GroupChat instance

        Returns:
            The next agent to speak
        """
        messages = groupchat.messages
        last_message = messages[-1] if messages else {}
        content = last_message.get("content", "").lower()

        logger.debug("Selecting next speaker after '%s'", last_speaker.name)

        # 1. Check for explicit routing in message
        if "next:" in content:
            # Extract agent name after "next:"
            try:
                agent_name = content.split("next:")[1].split()[0].strip()
                agent = registry.get_agent(agent_name)
                if agent:
                    logger.info("Explicit routing to '%s'", agent_name)
                    return agent
                logger.warning("Agent '%s' not found in registry", agent_name)
            except (IndexError, AttributeError) as exc:
                logger.debug("Failed to parse explicit routing: %s", exc)

        # 2. Check routing rules from registry
        routing_options = registry.get_routing_options(last_speaker.name)
        if routing_options:
            next_agent = routing_options[0]
            logger.info(
                "Following routing rule: '%s' -> '%s'",
                last_speaker.name,
                next_agent.name,
            )
            return next_agent

        # 3. Capability-based routing
        if any(
            keyword in content
            for keyword in ["research", "find", "gather", "investigate"]
        ):
            researchers = registry.find_agents_by_capability("research")
            if researchers and researchers[0] != last_speaker:
                logger.info("Routing to researcher based on content keywords")
                return researchers[0]

        if any(
            keyword in content
            for keyword in [
                "explain",
                "simplify",
                "teach",
                "clarify",
            ]
        ):
            explainers = registry.find_agents_by_capability("explanation")
            if explainers and explainers[0] != last_speaker:
                logger.info("Routing to explainer based on content keywords")
                return explainers[0]

        # 4. Follow default flow if provided
        if default_flow:
            try:
                current_idx = default_flow.index(last_speaker.name)
                next_idx = (current_idx + 1) % len(default_flow)
                next_agent_name = default_flow[next_idx]
                next_agent = registry.get_agent(next_agent_name)
                if next_agent:
                    logger.info(
                        "Following default flow: '%s' -> '%s'",
                        last_speaker.name,
                        next_agent_name,
                    )
                    return next_agent
            except (ValueError, IndexError):
                pass

        # 5. Fallback to round-robin through groupchat agents
        agents = groupchat.agents
        current_idx = agents.index(last_speaker)
        next_idx = (current_idx + 1) % len(agents)
        next_agent = agents[next_idx]

        logger.info(
            "Fallback round-robin: '%s' -> '%s'",
            last_speaker.name,
            next_agent.name,
        )

        return next_agent

    return select_speaker
