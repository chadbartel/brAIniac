# Standard Library
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# Third Party
import autogen
import requests

# Configure logging for more visibility (enable debug)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def wait_for_model_ready(
    base_url: str, max_retries: int = 30, retry_delay: int = 2
) -> bool:
    """
    Poll the server until the model is loaded and ready to serve requests.

    Args:
        base_url: The base URL of the llama server (e.g., http://localhost:8080/v1)
        max_retries: Maximum number of polling attempts
        retry_delay: Seconds to wait between retries

    Returns:
        True if model is ready, False if max retries exceeded
    """
    logging.info(
        "Polling server at %s to confirm model is loaded...", base_url
    )

    for attempt in range(1, max_retries + 1):
        try:
            # Try to get models list
            resp = requests.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    logging.info(
                        "✓ Model server ready! Found %d model(s): %s",
                        len(models),
                        [m.get("id") for m in models],
                    )
                    return True

            # Try a simple completion as fallback check
            payload: Dict[str, Any] = {
                "model": "local-model",
                "messages": [
                    {"role": "user", "content": "ping"},
                ],
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

        except requests.exceptions.RequestException as exc:
            logging.debug(
                "Attempt %d/%d failed: %s", attempt, max_retries, exc
            )

        if attempt < max_retries:
            logging.info(
                "Server not ready yet, retrying in %d seconds... (%d/%d)",
                retry_delay,
                attempt,
                max_retries,
            )
            time.sleep(retry_delay)

    logging.error(
        "✗ Failed to connect to model server after %d attempts", max_retries
    )
    return False


def check_local_server(base_url: str) -> None:
    """Probe the local Llama server for models and a quick ping."""
    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        logging.debug(
            "GET /models status=%s body=%s", resp.status_code, resp.text
        )
    except Exception as exc:  # pragma: no cover - diagnostic helper
        logging.warning("Failed to GET /models: %s", exc)

    try:
        payload: Dict[str, Any] = {
            "model": "local-model",
            "messages": [
                {
                    "role": "user",
                    "content": "Ping from main.py - are you alive?",
                }
            ],
            "temperature": 0.0,
            "max_tokens": 16,
        }
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10,
        )
        logging.debug(
            "POST /chat/completions status=%s body=%s",
            resp.status_code,
            resp.text,
        )
    except Exception as exc:  # pragma: no cover - diagnostic helper
        logging.warning("Failed to POST /chat/completions: %s", exc)


# Configuration for the local LLM
# Point this to your llama.cpp server's address and port
config_list = [
    {
        "model": "local-model",
        "base_url": "http://localhost:8080/v1",
        "api_key": "not-needed",
        # add price metadata to silence the "model not found" warning in autogen client
        "price": [0.0, 0.0],
    }
]

# General LLM configuration for the agents
llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

# 1. THE USER PROXY AGENT
# This agent acts as the user's proxy, executing code and gathering user input.
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human admin who will give the primary task. Interact with the other agents to get the task done.",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Set to True if you want to execute code in a Docker container
    },
    human_input_mode="TERMINATE",  # Pauses for human input before the conversation ends
    is_termination_msg=lambda x: x.get("content", "")
    .rstrip()
    .endswith("TERMINATE"),
)

# 2. WORKER AGENT TEMPLATES (STATIC VERSION)
researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message="You are a world-class research assistant. Your goal is to find comprehensive and accurate information on a given topic. After finding the information, report it back to the group.",
    llm_config=llm_config,
)

science_expert = autogen.AssistantAgent(
    name="Science_Expert",
    system_message="You are a world-leading expert in all scientific fields. You answer questions accurately based on your deep knowledge. When the research is done, you will provide the final answer.",
    llm_config=llm_config,
)

# 3. THE ORCHESTRATOR AGENT (GROUP CHAT MANAGER)
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, science_expert],
    messages=[],  # Initialize with an empty list for a new conversation
    max_round=12,
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
    Persist each agent's turn to a local file for post-mortem inspection.

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

    # Append to JSONL file (one JSON object per line)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn_data) + "\n")

    logging.info(
        "📝 Turn %d persisted: %s -> %s", turn_number, agent_name, filename
    )


def dump_conversation_state(log_dir: str = "logs") -> None:
    """
    Log and persist the full conversation state to a file.

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
    logging.debug(
        "Conversation state dump: %s", json.dumps(groupchat.messages, indent=2)
    )


# 4. INITIATE THE CONVERSATION
if __name__ == "__main__":
    base_url = config_list[0]["base_url"]

    # Wait for model to be loaded before starting conversation
    if not wait_for_model_ready(base_url, max_retries=30, retry_delay=2):
        logging.error("Model server is not ready. Exiting.")
        exit(1)

    # Optional: Additional diagnostics
    check_local_server(base_url)

    # Create a wrapper to capture each agent's turn
    original_initiate_chat = user_proxy.initiate_chat

    def wrapped_initiate_chat(*args, **kwargs):
        """Wrapper to persist turns during conversation"""
        result = original_initiate_chat(*args, **kwargs)

        # Persist each message after conversation completes
        for idx, msg in enumerate(groupchat.messages):
            agent_name = msg.get("name", "unknown")
            content = msg.get("content", "")
            persist_conversation_turn(agent_name, content, idx + 1)

        return result

    user_proxy.initiate_chat = wrapped_initiate_chat

    logging.info(
        "User_Proxy (to chat_manager):\n\nResearch the process of photosynthesis and then have the science expert explain it to me in simple terms.\n"
    )
    user_proxy.initiate_chat(
        orchestrator,
        message="Research the process of photosynthesis and then have the science expert explain it to me in simple terms.",
    )

    # Dump full conversation state at the end
    dump_conversation_state()
