import configparser
import json
from langchain_openai import ChatOpenAI 
from typing import Annotated
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition

# https://app.tavily.com/home?code=htf-dx2MTxCv2xWBJJDO38De2tyo2VbpCuRgruQ413U9n&state=eyJyZXR1cm5UbyI6Ii9ob21lIn0
# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot
# pip install langchain-openai
# pip install langgraph
# pip install langchain


def get_openai_keys():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['OpenAI_KEYS']

def TAVILY_API_KEY():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['TAVILY_API_KEY']

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
    openai_api_key=get_openai_keys()
)

# define a new tool that requests human assistance and then add it to the list of tools
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY()
tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}
 
def stream_graph_updates(user_input: BaseMessage, graph):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
        )
    
    for event in events :
        if "messages" in event:
            event["messages"][-1].pretty_print()
            

config = {"configurable": {"thread_id": "1"}}


def main():

    
    graph_builder = StateGraph(State)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # The `route_tools` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile(checkpointer=memory)
    
    while True:
        try:
            # I need some expert guidance for building an AI agent. Could you request assistance for me?
            user_input =  input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, graph)

            stream_graph_human_response(graph)

        except Exception as e:
            print("Error:", e)
            break

def stream_graph_human_response(graph):
    human_response = (
                "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
                " It's much more reliable and extensible than simple autonomous agents."
            )

    human_command = Command(resume={"data": human_response})
    events = graph.stream(human_command, config, stream_mode="values") #this line should take the control to tool
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


if __name__ == '__main__':
    main()