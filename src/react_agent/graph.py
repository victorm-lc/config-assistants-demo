"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""
from src.react_agent.tools import get_tools
from langgraph.prebuilt import create_react_agent
from src.utils import load_chat_model

from src.react_agent.configuration import Configuration
from langchain_core.runnables import RunnableConfig



async def make_graph(config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    
    # Get name from config or use default
    configurable = config.get("configurable", {}) if config else {}
    name = configurable.get("name", "react_agent")
    
    # Define a new graph
    llm = load_chat_model(configuration.model)
    tools = get_tools(config)
    prompt = configuration.system_prompt

    # Compile the builder into an executable graph
    # You can customize this by adding interrupt points for state updates
    graph = create_react_agent(
        model=llm, 
        tools=tools, 
        prompt=prompt, 
        config_schema=Configuration,
        name=name
    )

    return graph