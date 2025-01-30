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
from langgraph.checkpoint.memory import MemorySaver
from util import get_openai_keys, get_tavily_api_keys


llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=get_openai_keys())

os.environ["TAVILY_API_KEY"] = get_tavily_api_keys()
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

memory = MemorySaver()


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        # Initialize the BasicToolNode with a list of tools
        # Create a dictionary mapping tool names to tool instances for easy access
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        # Extract the list of messages from the inputs dictionary, or use an empty list if not present
        if messages := inputs.get("messages", []):
            # Get the last message from the list of messages
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: State):
    # return {"messages": [llm.invoke(state["messages"])]} //Call with out tools
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def stream_graph_updates(user_input: BaseMessage, graph):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()


def addition(state: State):
    messages = state["messages"]
    if len(messages) == 1:
        try:
            number = float(messages[-1].content)
            return {"messages": [SystemMessage(content=f"The result is {5 + number}.")]}
        except ValueError:
            return {
                "messages": [SystemMessage(content="Please provide a valid number.")]
            }
    else:
        return {"messages": [SystemMessage(content="I don't understand.")]}


def multiply(state: State):
    messages = state["messages"]
    if len(messages) == 1:
        try:
            number = float(messages[-1].content)
            return {"messages": [SystemMessage(content=f"The result is {5 * number}.")]}
        except ValueError:
            return {
                "messages": [SystemMessage(content="Please provide a valid number.")]
            }
    else:
        return {"messages": [SystemMessage(content="I don't understand.")]}


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


config = {"configurable": {"thread_id": "1"}}


def main():

    graph_builder = StateGraph(State)
    tool_node = BasicToolNode(tools=[tool])

    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)
    # graph_builder.add_node("add", addition)
    # graph_builder.add_node("multiply", multiply)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # The `route_tools` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile(checkpointer=memory)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, graph)
        except Exception as e:
            print("Error:", e)
            break


if __name__ == "__main__":
    main()
