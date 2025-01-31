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
from util import create_and_save_gaph_image, get_openai_keys, get_tavily_api_keys

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=get_openai_keys())


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    print("\n\033[92mRequesting human assistance, going to wait here till get a resume call for:\033[0m", query)
    # This intrupt will send the query to the client
    human_response = interrupt({"prompt": query})
    print("\n\033[91mHuman response:\033[0m", human_response)
    return human_response["data"]

os.environ["TAVILY_API_KEY"] = get_tavily_api_keys()
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


config = {"configurable": {"thread_id": "1"}}

 # To Test type this prompt in the message:
 # I need some expert guidance for building an AI agent. Could you request assistance for me?
 # in response Type this for human assistance:
 #{"data": "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents."}
def human_in_loop():
    graph_builder = StateGraph(State)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile(checkpointer=memory)
    return graph
     






if __name__ == "__main__":
    human_in_loop()
