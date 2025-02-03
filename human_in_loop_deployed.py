# https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/#agent
# Human in loop example. This example demonstrates how to create a LangGraph that waits for user input before continuing.
# Update the Langgraph.json file.
# To run the code :
#   Langgraph dev 
#  Then run the code in Langraph client directory which will act as client for the code.


from langchain_openai import ChatOpenAI
from typing import Annotated
from langchain_core.tools import tool
from openai import BaseModel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode
from util import get_openai_keys, get_tavily_api_keys

model = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=get_openai_keys())

class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # If tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human input
    # For example, send a slack message, etc
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "action"

# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]} # We return a list, because this will get added to the existing list of messages

# We define a fake node to ask the human

def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    #In client provide the following input to get the response from the human
    # {"data": "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents."}
    content = interrupt(state["messages"][-1].content)
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": content}]
    return {"messages": tool_message}


config = {"configurable": {"thread_id": "1"}}

@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return f"I looked up: {query}. Result: It's sunny in {query}, but you better look out if you're a Gemini 😈."


 # To Test type this prompt in the message:
 # I need some expert guidance for building an AI agent. Could you request assistance for me?
 # in response Type this for human assistance:
 #{"data": "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents."}


tools = [search]
tool_node = ToolNode(tools)
model = model.bind_tools(tools + [AskHuman])
memory = MemorySaver()

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
)
workflow.add_edge("action", "agent")
workflow.add_edge("ask_human", "agent")
app = workflow.compile(checkpointer=memory)



 
