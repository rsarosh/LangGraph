# Basic Chatbot
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal
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

config = {"configurable": {"thread_id": "1"}}


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

 
def add(state: State):
    """ Adds two numbers together. """
    state["messages"].append(HumanMessage(content="This is addition"))
    return state
    
def devide(state: State):
    """ Divides two numbers. """
    state["messages"].append(HumanMessage(content="This is devision"))
    return state
    
def multiply(state: State):
    """ Multiplies two numbers together. """
    state["messages"].append(HumanMessage(content="This is multiplication"))
    return state

def router_function(state: State) -> Literal["add", "multiply", "devide", "router_node"]:
    if state["messages"][-1].content == "add":
        return "add"
    elif state["messages"][-1].content == "multiply":
        return "multiply"
    elif state["messages"][-1].content == "devide":
        return "devide"
    else:
        return "router_node"
    
    
    
def stream_graph_updates(user_input: BaseMessage, graph):
   
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for value in event.values():
                if value is not None:
                    if value["messages"][-1].content is not None:
                        print("Assistant:", value["messages"][-1].content)


def router_node(state: State) :
    """ Just a router function that routes to different functions. """
    print("This is router node visit")
    # state["messages"].append(HumanMessage(content="This is router"))
    return state

def main():
    graph_builder = StateGraph(State)
    # Define Nodes
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("add", add)
    graph_builder.add_node("multiply", multiply)
    graph_builder.add_node("devide", devide)
    # Define Edges
    graph_builder.add_edge(START, "router_node")
    graph_builder.add_edge("router_node", END)
    graph_builder.add_edge("add", END)
    graph_builder.add_edge("multiply", END)
    graph_builder.add_edge("devide", END)
    graph_builder.add_conditional_edges("router_node", router_function)
    graph = graph_builder.compile()
    create_and_save_gaph_image(graph, filename="intro_to_langgraph_router.png")
    print("Welcome to the introduction to langgraph router! Type 'quit' to exit.")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
        except Exception as e:
            print("Error:", e)
            break
       
        # _input: State = State (
        #     messages = [BaseMessage(content=user_input)]
        #     )
        # graph.invoke(input = _input)
        stream_graph_updates(user_input, graph)
if __name__ == '__main__':
    main()
