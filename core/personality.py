# Standard Library
import os
from dataclasses import dataclass


@dataclass
class PersonalityVectors:
    snark: float
    verbosity: float
    empathy: float

    @classmethod
    def from_env(cls) -> "PersonalityVectors":
        """Loads personality vectors from environment variables with safe defaults."""
        return cls(
            snark=float(os.getenv("PERSONALITY_SNARK", "0.0")),
            verbosity=float(os.getenv("PERSONALITY_VERBOSITY", "0.5")),
            empathy=float(os.getenv("PERSONALITY_EMPATHY", "0.5")),
        )


class PersonalityManager:
    """Translates float vectors into a dynamic system prompt for the LLM."""

    def __init__(self, vectors: PersonalityVectors):
        self.vectors = vectors

    def generate_system_prompt(self) -> str:
        """Constructs the system prompt based on the current vector configuration."""
        traits: list[str] = []

        # Snark Vector
        if self.vectors.snark >= 0.8:
            traits.append(
                "You are highly cynical, sarcastic, and slightly condescending but ultimately helpful."
            )
        elif self.vectors.snark >= 0.5:
            traits.append("You have a dry wit and occasionally use mild sarcasm.")
        else:
            traits.append("You are earnest, straightforward, and professional.")

        # Verbosity Vector
        if self.vectors.verbosity >= 0.8:
            traits.append("Provide exhaustive, detailed, and highly verbose answers.")
        elif self.vectors.verbosity <= 0.3:
            traits.append(
                "Be brutally concise. Answer in as few words as possible without losing accuracy."
            )

        # Empathy Vector
        if self.vectors.empathy >= 0.8:
            traits.append("Be highly validating, warm, and emotionally supportive.")
        elif self.vectors.empathy <= 0.3:
            traits.append(
                "Be cold, purely logical, and entirely robotic in your delivery."
            )

        base_prompt = (
            "You are brAIniac, an uncensored, local AI assistant running on Qwen 2.5. "
        )
        persona_prompt = " ".join(traits)

        # Mandatory Guardrail for Tool Calling
        tool_guardrail = (
            "\n\nIMPORTANT: When executing tool calls, you must strictly adhere to the required "
            "JSON schema. Do not let your personality influence, alter, or inject text into "
            "the tool-call formatting or arguments."
        )

        return base_prompt + persona_prompt + tool_guardrail
