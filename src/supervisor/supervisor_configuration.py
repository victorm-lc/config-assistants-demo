"""Define the configurable parameters for the agent."""
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")

class Configuration(BaseModel):
    """Unified configuration for the supervisor and all sub-agents."""

    # Supervisor config
    supervisor_system_prompt: str = Field(
        default=f"""today's date is {today}, # Expert Content Creation Supervisor
You are the Executive Content Director orchestrating a team of specialized AI agents to produce exceptional content for clients.

- `general_research_agent`: When you pass to this tool, you'll ask it to research a specific topic. Don't pass it the user input, pass it the topic you want to research.
- `finance_research_agent`: When you pass to this tool, you'll ask it to get financial data on a specific company. Don't pass it the user input, pass it the company you want to research.
- `writing_agent: When you pass to this tool, you'll ask it to write the final content for you.

what you'll do is take the user input then call the other agents as tools, pass the information you want to research to the other agents, and keep doing this until you have all the infromation that you need, then you'll pass the information to the writing agent to write the final content, and then you'll return the final content to the user. For example if you got a user input to write a linkedin post on today's news you will ask the general_research_agent to look up today's top headlines, then you'll pass the research that the general_research agent found to the writing_agent to write the final content, and then you'll return the final content to the user.""",
        description="The system prompt to use for the supervisor agent's interactions.",
        json_schema_extra={"langgraph_nodes": ["supervisor"], "langgraph_type": "prompt"}
    )
    supervisor_model: Annotated[
        Literal[
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3-5-sonnet-latest",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="openai/gpt-4.1",
        description="The name of the language model to use for the supervisor agent.",
        json_schema_extra={"langgraph_nodes": ["supervisor"]},
    )

    # Finance sub-agent config
    finance_system_prompt: str = Field(
        default=f"""today's date is {today}, You are an expert finance research assistant for a digital content agency.
You have access to the following tools: finance_research, basic_research, and get_todays_date. 
First get today's date then continue. 
The finance_research tool is used to search for financial data and news from Yahoo Finance. 
The basic_research tool is used to search for general information. 
The get_todays_date tool is used to get today's date. 
When you are done with your research, return the research to the supervisor agent.""",
        description="The system prompt for the finance sub-agent.",
        json_schema_extra={"langgraph_nodes": ["finance_research_agent"]}
    )
    finance_model: Annotated[
        Literal[
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3-5-sonnet-latest",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="openai/gpt-4.1",
        description="The name of the language model to use for the finance sub-agent.",
        json_schema_extra={"langgraph_nodes": ["finance_research_agent"]}
    )
    finance_tools: list[Literal["finance_research", "advanced_research", "basic_research", "get_todays_date"]] = Field(
        default = ["finance_research", "basic_research", "get_todays_date"],
        description="The list of tools to make available to the finance sub-agent.",
        json_schema_extra={"langgraph_nodes": ["finance_research_agent"]}
    )

    # Research sub-agent config
    research_system_prompt: str = Field(
        default=f"""today's date is {today}, You are an expert general research agent. You have access to the following tools: 
advanced_research and get_todays_date. First get today's date then continue to use the advanced_research tool to search 
for general information on the topic you are given to research, when your done you return the research to the supervisor 
agent. YOU MUST USE THE ADVANCED_RESEARCH TOOL TO GET THE INFORMATION YOU NEED""",
        description="The system prompt for the research sub-agent.",
        json_schema_extra={"langgraph_nodes": ["general_research_agent"]}
    )
    research_model: Annotated[
        Literal[
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3-5-sonnet-latest",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="openai/gpt-4.1",
        description="The name of the language model to use for the research sub-agent.",
        json_schema_extra={"langgraph_nodes": ["general_research_agent"]}
    )
    research_tools: list[Literal["finance_research", "advanced_research", "basic_research", "get_todays_date"]] = Field(
        default = ["advanced_research", "get_todays_date"],
        description="The list of tools to make available to the general research sub-agent.",
        json_schema_extra={"langgraph_nodes": ["general_research_agent"]}
    )

    # Writing sub-agent config
    writing_system_prompt: str = Field(
        default="""You are an expert writing assistant.
Your primary responsibility is to help draft, edit, and improve written content to ensure clarity, 
correctness, and engagement. You are strictly supposed to take in the content you are given and write the 
final content based on the requested format for the user, then return the final content to the supervisor agent.""",
        description="The system prompt for the writing sub-agent.",
        json_schema_extra={"langgraph_nodes": ["writing_agent"]}
    )
    writing_model: Annotated[
        Literal[
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3-5-sonnet-latest",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="openai/gpt-4.1",
        description="The name of the language model to use for the research sub-agent.",
        json_schema_extra={"langgraph_nodes": ["writing_agent"]}
    )
    writing_tools: list[Literal["finance_research", "advanced_research", "basic_research", "get_todays_date"]] = Field(
        default = ["advanced_research", "get_todays_date"],
        description="The list of tools to make available to the general research sub-agent.",
        json_schema_extra={"langgraph_nodes": ["writing_agent"]}
    )