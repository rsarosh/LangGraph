# Basic Chatbot
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import os
from util import create_and_save_gaph_image, get_openai_keys, get_tavily_api_keys, save_image


llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
    openai_api_key=get_openai_keys()
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


@tool
def add(x: int, y: int):
    """ Adds two numbers together. """
    return x + y
    

@tool
def devide(state: State, x: int, y: int):
    """ Divides two numbers. """
    return x / y
    

@tool
def multiply(x: int, y: int):
    """ Multiplies two numbers together. """
    return x * y
    

tools = [add, multiply, devide]
llm_with_tools = llm.bind_tools(tools)


def stream_graph_updates(user_input: BaseMessage, graph):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
        
def main():
    graph_builder = StateGraph(State)
    # Define Nodes
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    # Define Edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("chatbot", END)
    # Define Conditional Edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph = graph_builder.compile()
    create_and_save_gaph_image(graph, filename="intro_to_langgraph_tool.png")
    print("Welcome to the introduction to langgraph! Type 'quit' to exit.")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
        except Exception as e:
            print("Error:", e)
            break

        stream_graph_updates(user_input, graph)
if __name__ == '__main__':
    main()
