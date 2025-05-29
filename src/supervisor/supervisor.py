from langchain_core.runnables import RunnableConfig
from supervisor.supervisor_configuration import Configuration
from src.react_agent.utils import load_chat_model
from src.react_agent.graph import make_graph as make_react_graph

from langgraph_supervisor import create_supervisor

# Helper function to create subagent graphs
async def make_subagent_graph(config: RunnableConfig, agent_type: str, name: str):
    """Create a subagent graph with the correct configuration mapping.
    
    Args:
        config: The RunnableConfig from the supervisor
        agent_type: The type of agent to create ("research", "finance", or "writing")
        name: The name to give the graph
        
    Returns:
        A compiled graph for the subagent
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Map from supervisor config structure to react agent config structure
    if agent_type == "research":
        configurable = {
            "system_prompt": configuration.research_system_prompt,
            "model": configuration.research_model,
            "selected_tools": configuration.research_tools
        }
    elif agent_type == "finance":
        configurable = {
            "system_prompt": configuration.finance_system_prompt,
            "model": configuration.finance_model,
            "selected_tools": configuration.finance_tools
        }
    elif agent_type == "writing":
        configurable = {
            "system_prompt": configuration.writing_system_prompt,
            "model": configuration.writing_model,
            "selected_tools": configuration.writing_tools
        }
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    # Create a new config with the mapped configurable values
    subagent_config = {**config, "configurable": configurable}
    
    # Use the existing make_graph function from react_agent
    return await make_react_graph(subagent_config, name)

# Main graph construction
async def make_supervisor_graph(config: RunnableConfig):
    # Parse unified config
    configuration = Configuration.from_runnable_config(config)

    # Create the subagent graphs
    finance_graph = await make_subagent_graph(config, "finance", "finance_agent")
    research_graph = await make_subagent_graph(config, "research", "general_research_agent")
    writing_graph = await make_subagent_graph(config, "writing", "writing_agent")

    workflow = create_supervisor(
        agents = [finance_graph, research_graph, writing_graph],
        model=load_chat_model(configuration.supervisor_model),
        prompt=configuration.supervisor_system_prompt,
        config_schema = Configuration
    )

    return workflow.compile()
