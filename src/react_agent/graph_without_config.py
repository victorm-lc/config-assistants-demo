"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""
from react_agent.tools import basic_research_tool, get_todays_date
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model


async def make_graph():
    
    # initialize our model and tools    
    llm = init_chat_model("openai/gpt-4.1-mini")
    tools = [basic_research_tool, get_todays_date]
    prompt = "You are a helpful AI assistant!"

    # Compile the builder into an executable graph
    graph = create_react_agent(
        model = llm, 
        tools = tools, 
        prompt=prompt
    )

    return graph