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

# Configure logging at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def wait_for_model_ready(
    base_url: str, max_retries: int = 30, retry_delay: int = 2
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
        "Polling server at %s to confirm model is loaded...", base_url
    )

    for attempt in range(1, max_retries + 1):
        try:
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

    logging.error(
        "✗ Failed to connect after %d attempts", max_retries
    )
    return False


# Configuration for the local LLM with stop tokens
config_list = [
    {
        "model": "local-model",
        "base_url": "http://localhost:8080/v1",
        "api_key": "not-needed",
        "price": [0.0, 0.0],
    }
]

# LLM configuration with stop sequences to prevent template tokens
llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "max_tokens": 500,
    "stop": ["<|im_end|>", "<|im_start|>"],
}

# User proxy with better termination logic
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message=(
        "You are a human user. You give tasks to the team and "
        "respond when needed. Reply TERMINATE when satisfied."
    ),
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda x: (
        x.get("content", "")
        .rstrip()
        .upper()
        .endswith("TERMINATE")
    ),
)

researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message=(
        "You are a research assistant. Find information on topics "
        "and report back concisely. When done, say 'Research "
        "complete.'"
    ),
    llm_config=llm_config,
)

science_expert = autogen.AssistantAgent(
    name="Science_Expert",
    system_message=(
        "You are a science expert. Explain topics in simple terms. "
        "When you have answered the question, say 'TERMINATE'."
    ),
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, science_expert],
    messages=[],
    max_round=8,
    speaker_selection_method="round_robin",
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

    logging.info(
        "📝 Turn %d: %s", turn_number, agent_name
    )


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

    if not wait_for_model_ready(base_url, max_retries=30, retry_delay=2):
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

    logging.info(
        "Starting conversation: Research photosynthesis\n"
    )
    
    user_proxy.initiate_chat(
        orchestrator,
        message=(
            "Research the process of photosynthesis and have the "
            "science expert explain it to me in simple terms."
        ),
    )

    dump_conversation_state()
    logging.info("✓ Conversation complete")