"""Test script for local Llama AI server."""

from openai import OpenAI


def test_local_llm_server() -> None:
    """Test the local LLM server by sending a chat completion request.

    This function creates an OpenAI client pointing to the local server
    and sends a simple chat message to verify the server is working.
    """
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
    )

    print("Sending request to local LLM server...")

    try:
        completion = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            temperature=0.7,
            max_tokens=100,
            stop=["<|im_end|>", "<|im_start|>", "\n\n"],
        )

        print("\nResponse from server:")
        print(completion.choices[0].message.content)

    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    test_local_llm_server()
