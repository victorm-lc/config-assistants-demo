"""Define the configurable parameters for the agent."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """The configuration for the agent."""

    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="The system prompt to use for the agent's interactions. "
        "This prompt sets the context and behavior for the agent."
    )

    model: Annotated[
            Literal[
                "anthropic/claude-3-7-sonnet-latest",
                "anthropic/claude-3-5-sonnet-latest",
                "openai/gpt-4.1",
                "openai/gpt-4.1-mini"
            ],
            {"__template_metadata__": {"kind": "llm"}},
        ] = Field(
            default="anthropic/claude-3-5-sonnet-latest",
            description="The name of the language model to use for the agent's main interactions. "
        "Should be in the form: provider/model-name."
    )

    selected_tools: list[Literal["finance_research", "advanced_research", "basic_research", "get_todays_date"]] = Field(
        default = ["get_todays_date"],
        description="The list of tools to use for the agent's interactions. "
        "This list should contain the names of the tools to use."
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config_dict = ensure_config(config)
        configurable = config_dict.get("configurable") or {}
        # Use model_fields instead of fields() in Pydantic v2
        return cls(**{k: v for k, v in configurable.items() if k in cls.model_fields})