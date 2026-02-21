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

# Configure logging at DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Define prompts with explicit instructions and examples
USER_PROXY_PROMPT = """
A human admin who will give the primary task. Interact with the other agents to get the task done.
The user's task must be completed.
"""

# Researcher prompt with few-shot examples for the specific model
RESEARCHER_PROMPT = """
You are a COMPREHENSIVE research assistant with web search tools. Your job is to:
1. ALWAYS start with the get_current_date() tool to know today's date
2. Use search_web() multiple times with different queries to get comprehensive results
3. Provide detailed synthesis of all findings in a clear format
4. End with "NEXT: Political_Expert" (exact text)

CRITICAL: You MUST use the tools. DO NOT say "I don't know" or "I'll search" - just use the tools directly.

EXAMPLE OF HOW YOU SHOULD RESPOND:

To research a topic like "Trump latest news":

Step 1: Use get_current_date()
***** Suggested tool call: get_current_date *****
Arguments: {}
******************************************************************************

After getting date: "Today is October 22, 2025. I'll search for latest Trump news."

Step 2: Use search_web() with specific queries
***** Suggested tool call: search_web *****
Arguments: {"query":"Trump latest news October 2025"}
******************************************************************************

Step 3: Search with variations
***** Suggested tool call: search_web *****
Arguments: {"query":"President Trump recent developments 2025"}
******************************************************************************

Step 4: Provide comprehensive report with specific details from searches
"Based on my research from October 22, 2025, Trump has [specific findings from search results]..."

Step 5: End with routing phrase
"NEXT: Political_Expert"

FOLLOW THIS EXACT PATTERN - USE THE TOOLS!
"""

# Political Expert prompt with clear ending instruction
POLITICAL_EXPERT_PROMPT = """
You are a political educator who explains complex political topics in simple terms.

INSTRUCTIONS:
1. When the Researcher provides information about a political topic or figure, explain it in 2-3 simple paragraphs
2. Focus on: What happened? Why is it important? What might happen next?
3. Use analogies to make complex ideas easier to understand
4. ALWAYS end your message with the exact text: "TERMINATE"

IMPORTANT: Do not ask follow-up questions or be conversational. Your job is to explain and end with TERMINATE.

EXAMPLE RESPONSE:
"George Santos was a congressman who lied about many things, including his work history and education. This matters because elected officials are supposed to be truthful and represent their constituents honestly.

The recent developments show that [explanation of the latest news]. This could lead to [potential consequences or outcomes].

TERMINATE"
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

# LLM configuration with higher temperature for more decisive responses
llm_config = {
    "config_list": config_list,
    "temperature": 0.8,  # Increased from 0.7 for more decisive outputs
}

# Initialize the agent registry
registry = AgentRegistry(llm_config)


# Enhanced termination detection for more reliable conversation ending
def is_termination_msg(message: Dict[str, Any]) -> bool:
    """
    Checks if a message indicates conversation termination.

    Args:
        message: The message to check

    Returns:
        True if the message should terminate the conversation
    """
    if not message.get("content"):
        return False

    content = message.get("content", "").upper()
    return (
        "TERMINATE" in content
        or
        # Also check for partial completions that suggest termination
        (content.endswith("TERM") and len(content) > 10)
    )


# Register the User Proxy with tool execution capability
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message=USER_PROXY_PROMPT,
    code_execution_config=False,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=15,
    is_termination_msg=is_termination_msg,
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

# Register the Political Expert agent (fixed name)
political_expert = registry.register_agent(
    name="Political_Expert",  # Note: fixed spelling from "Policital_Expert"
    system_message=POLITICAL_EXPERT_PROMPT,
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
    agents=[user_proxy, researcher, political_expert],
    messages=[],
    max_round=25,
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

    logging.info("Starting conversation: Research George Santos\n")

    user_proxy.initiate_chat(
        orchestrator,
        message=(
            "Research George Santos. I need current information about:\n"
            "1. Who is he and what crimes did he commit?\n"
            "2. What was his sentence?\n"
            "3. What's the most recent development in 2025?\n"
            "4. Is he currently in prison?\n\n"
            "Make sure to use search tools to find recent 2025 information."
        ),
    )

    dump_conversation_state()
    logging.info("✓ Conversation complete")
