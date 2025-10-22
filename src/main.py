# Standard Library
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    # Standard Library
    import io

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

# Third Party
import autogen
import requests
from dotenv import load_dotenv

load_dotenv()

# My Modules
from agent_registry import AgentRegistry, create_dynamic_selector
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

# Configure logging at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Define prompts
USER_PROXY_PROMPT = """
A human admin who will give the primary task. Interact with the other agents to get the task done.
The user's task must be completed.
"""

RESEARCHER_PROMPT = """
You are a master researcher. Your role is to take a user's request, break it down into searchable questions, and use the tools at your disposal to find the most accurate and up-to-date information.

**Your Process:**
1.  **Deconstruct:** Break down the user's complex request into a series of smaller, specific questions that can be answered with web searches.
2.  **Search:** Use the `search_web` tool for each question to find relevant articles and sources.
3.  **Scrape & Analyze:** For the most promising URLs from your search, use the `scrape_url` tool to read the full content.
4.  **Synthesize:** After gathering all the information, compile a comprehensive report that directly answers the user's original request. Address each point of the request clearly.
5.  **Final Output:** Present your final, synthesized report to the group. Do not ask for the next step. Simply provide the completed research.
"""


def wait_for_model_ready(
    base_url: str,
    max_retries: int = 30,
    retry_delay: int = 2,
) -> bool:
    """
    Poll the server until the model is loaded and ready.

    Args:
        base_url: The base URL of the llama server
        max_retries: Maximum number of polling attempts
        retry_delay: Seconds to wait between retries

    Returns:
        True if model is ready, False if max retries exceeded
    """
    logging.info(
        "Polling server at %s to confirm model is loaded...",
        base_url,
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    model_ids = [m.get("id") for m in models]
                    logging.info(
                        "✓ Model server ready! Found %d model(s): %s",
                        len(models),
                        model_ids,
                    )
                    return True

            payload: Dict[str, Any] = {
                "model": "local-model",
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.0,
                "max_tokens": 5,
            }
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10,
            )
            if resp.status_code == 200:
                logging.info(
                    "✓ Model server ready! Completion endpoint responding."
                )
                return True

        except requests.exceptions.RequestException:
            pass

        if attempt < max_retries:
            logging.info(
                "Server not ready, retrying in %d seconds... (%d/%d)",
                retry_delay,
                attempt,
                max_retries,
            )
            time.sleep(retry_delay)

    logging.error("✗ Failed to connect after %d attempts", max_retries)
    return False


# Configuration for the local LLM
config_list = [
    {
        "model": "local-model",
        "base_url": "http://localhost:8080/v1",
        "api_key": "not-needed",
        "price": [0.0, 0.0],
    }
]

# LLM configuration
llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

# Initialize the agent registry
registry = AgentRegistry(llm_config)

# Register the User Proxy with tool execution capability
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message=USER_PROXY_PROMPT,
    code_execution_config=False,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=15,
    is_termination_msg=lambda x: (
        x.get("content", "") and ("TERMINATE" in x.get("content", "").upper())
    ),
    function_map=TOOL_FUNCTIONS,
)

# Register the Researcher agent with web search tools
researcher = registry.register_agent(
    name="Researcher",
    system_message=RESEARCHER_PROMPT,
    capabilities=["research", "data_gathering", "analysis"],
    can_route_to=["User_Proxy", "Political_Expert"],
    tools=TOOL_DEFINITIONS,
)

# Register the Science Expert agent
science_expert = registry.register_agent(
    name="Policital_Expert",
    system_message=(
        "You are a political educator. When the Researcher provides "
        "technical information:\n"
        "1. Explain the concepts in simple, easy-to-understand terms\n"
        "2. Use analogies and examples when helpful\n"
        "3. End your explanation with: TERMINATE\n"
        "This completes the conversation."
    ),
    capabilities=["explanation", "simplification", "teaching"],
    can_route_to=["User_Proxy"],
)

# Create dynamic speaker selector with default flow
speaker_selector = create_dynamic_selector(
    registry=registry,
    default_flow=["User_Proxy", "Researcher", "Political_Expert"],
)

# Create the group chat with dynamic speaker selection
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, science_expert],
    messages=[],
    max_round=15,
    speaker_selection_method=speaker_selector,
    allow_repeat_speaker=False,
)

orchestrator = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)


def persist_conversation_turn(
    agent_name: str,
    message_content: str,
    turn_number: int,
    log_dir: str = "logs",
) -> None:
    """
    Persist each agent turn to a JSONL file.

    Args:
        agent_name: Name of the agent sending the message
        message_content: The message content
        turn_number: The turn number in the conversation
        log_dir: Directory to store log files
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = log_path / f"conversation_{timestamp}.jsonl"

    turn_data = {
        "turn": turn_number,
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "message": message_content,
    }

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn_data) + "\n")

    logging.info("📝 Turn %d: %s", turn_number, agent_name)


def dump_conversation_state(log_dir: str = "logs") -> None:
    """
    Save the full conversation state to a JSON file.

    Args:
        log_dir: Directory to store the conversation log
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = log_path / f"full_conversation_{timestamp}.json"

    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": groupchat.messages,
        "total_turns": len(groupchat.messages),
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2)

    logging.info("💾 Full conversation saved to: %s", filename)


if __name__ == "__main__":
    base_url = config_list[0]["base_url"]

    if not wait_for_model_ready(
        base_url,
        max_retries=30,
        retry_delay=2,
    ):
        logging.error("Model server not ready. Exiting.")
        exit(1)

    original_initiate_chat = user_proxy.initiate_chat

    def wrapped_initiate_chat(*args, **kwargs):
        """Wrapper to persist turns during conversation."""
        result = original_initiate_chat(*args, **kwargs)

        for idx, msg in enumerate(groupchat.messages):
            agent_name = msg.get("name", "unknown")
            content = msg.get("content", "")
            persist_conversation_turn(agent_name, content, idx + 1)

        return result

    user_proxy.initiate_chat = wrapped_initiate_chat

    logging.info("Starting conversation: Research Geoerge Santos\n")

    user_proxy.initiate_chat(
        orchestrator,
        message=(
            "Research the current political scandal surrounding George Santos and "
            "have the political expert explain it to me in simple terms. "
            "What was the most recent development? Is he still in jail?"
        ),
    )

    dump_conversation_state()
    logging.info("✓ Conversation complete")
