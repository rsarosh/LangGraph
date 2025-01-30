# Tools Chatbot with ToolNode and tools_condition
import configparser
import json
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langgraph.prebuilt import ToolNode, tools_condition
from util import get_openai_keys, get_tavily_api_keys

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=get_openai_keys())

os.environ["TAVILY_API_KEY"] = get_tavily_api_keys()
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def stream_graph_updates(user_input: BaseMessage, graph):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def main():
    graph_builder = StateGraph(State)
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile()

    while True:
        try:
            # Tell me about BA company and its Stock and its price
            # user_input =  input("User: ")
            user_input = "Tell me about stock BA and its Stock and its price "
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, graph)
            break
        except Exception as e:
            print("Error:", e)
            break


if __name__ == "__main__":
    main()
