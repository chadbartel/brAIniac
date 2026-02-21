"""
Dynamic agent registry for runtime agent management.

This module provides a registry pattern for managing agents dynamically,
including capability-based routing and on-demand agent creation.
"""

# Standard Library
import re
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
        Select the next speaker based on conversation state.

        Args:
            last_speaker: The agent who spoke last
            groupchat: The group chat instance

        Returns:
            The next agent to speak
        """
        # Get the last speaker name
        last_speaker_name = getattr(last_speaker, "name", "")

        # Extract content from the last message for content-based routing
        last_message = groupchat.messages[-1] if groupchat.messages else {}
        content = (
            last_message.get("content", "").lower()
            if isinstance(last_message.get("content"), str)
            else ""
        )

        # Check for tool calls to route to User_Proxy
        if last_message.get("tool_calls"):
            logger.info(
                f"Routing tool call from '{last_speaker_name}' to 'User_Proxy' for execution"
            )
            return next(a for a in groupchat.agents if a.name == "User_Proxy")

        # Check for tool results being returned to the original caller
        if (
            last_speaker_name == "User_Proxy"
            and last_message.get("tool_call_id")
            and len(groupchat.messages) >= 3
        ):

            # Find the agent who made the tool call
            for i in range(len(groupchat.messages) - 2, -1, -1):
                msg = groupchat.messages[i]
                if msg.get("tool_calls") and any(
                    tc.get("id") == last_message.get("tool_call_id")
                    for tc in msg.get("tool_calls", [])
                ):
                    caller_name = msg.get("name")
                    if caller_name and caller_name in registry._agents:
                        logger.info(
                            f"Returning tool results from 'User_Proxy' to '{caller_name}'"
                        )
                        return registry._agents[caller_name]

        # Check for "NEXT:" routing directives in the message content
        if "next:" in content:
            # Extract agent name after "NEXT:"
            next_agent_pattern = r"next:\s*(\w+)"
            match = re.search(next_agent_pattern, content, re.IGNORECASE)
            if match:
                next_agent_name = match.group(1).strip()
                # Case-insensitive matching for agent names
                for name in registry._agents:
                    if name.lower() == next_agent_name.lower():
                        logger.info(
                            f"Following routing directive: '{last_speaker_name}' -> '{name}'"
                        )
                        return registry._agents[name]
                logger.warning(
                    f"Agent '{next_agent_name}' not found in registry"
                )

        # Check for explicit routing rules
        if last_speaker_name in registry._routing_rules:
            for target_name in registry._routing_rules[last_speaker_name]:
                if target_name in registry._agents:
                    logger.info(
                        f"Following routing rule: '{last_speaker_name}' -> '{target_name}'"
                    )
                    return registry._agents[target_name]

        # Check for keyword-based routing
        if last_speaker_name == "User_Proxy" and content:
            research_keywords = [
                "research",
                "find",
                "search",
                "look up",
                "information",
                "data",
                "study",
                "investigate",
            ]

            if any(keyword in content for keyword in research_keywords):
                logger.info("Routing to researcher based on content keywords")
                return registry._agents.get("Researcher")

        # Follow default flow if defined
        if default_flow and last_speaker_name in default_flow:
            idx = default_flow.index(last_speaker_name)
            if idx < len(default_flow) - 1:
                next_name = default_flow[idx + 1]
                if next_name in registry._agents:
                    logger.info(
                        f"Following default flow: '{last_speaker_name}' -> '{next_name}'"
                    )
                    return registry._agents[next_name]

        # Round-robin fallback (pick next agent)
        agents = groupchat.agents
        if len(agents) > 1:
            last_idx = next(
                (
                    i
                    for i, a in enumerate(agents)
                    if a.name == last_speaker_name
                ),
                -1,
            )
            next_idx = (last_idx + 1) % len(agents)
            logger.info(
                f"Fallback round-robin: '{last_speaker_name}' -> '{agents[next_idx].name}'"
            )
            return agents[next_idx]

        # Default to first agent
        logger.warning(
            f"No suitable next speaker found after '{last_speaker_name}'"
        )
        return agents[0]

    return select_speaker
