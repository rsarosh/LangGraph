# Basic Chatbot
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from util import get_openai_keys, get_tavily_api_keys
from IPython.display import Image, display


llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
    openai_api_key=get_openai_keys()
)
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def stream_graph_updates(user_input: BaseMessage, graph):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def main():

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()
    image = Image(graph.get_graph().draw_mermaid_png())
    save_image(image)
    print("Welcome to the basic chatbot! Type 'quit' to exit.")
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

def save_image(image):
    image_path = "graph_image.png"
    with open(image_path, "wb") as f:
        f.write(image.data)


if __name__ == '__main__':
    main()
