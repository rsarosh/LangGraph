import time
import pandas as pd
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from util import get_anthropic_keys, get_deepseek_keys, get_openai_keys, get_pinecone_keys
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains.base import Chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from uuid import uuid4


model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=get_openai_keys()
)

# model = ChatAnthropic(
#     model="claude-3-5-sonnet-20240620",
#     api_key=get_anthropic_keys())


def call_llm():
    messages = [
        HumanMessage(content="What is the capital of France?"),
        SystemMessage(content="You are an expert of genral knowlege."),
    ]
    response = model.invoke(messages)
    print("Response from OpenAI:", response)


def call_with_prompt():
    prompt = setup_prompt()
    concept = "pydantic"  # inheritance
    response = model.invoke(prompt.format(concept=concept))
    print("Response from OpenAI:", response)


def setup_prompt(concept=""):
    template = """ You are an expert of python.  explain the concept of {concept}"""
    prompt = PromptTemplate(
        input_variables=["concept"],
        template=template
    )
    return prompt


def call_chain():
    prompt = setup_prompt()
    concept = "inheritance"
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({'concept': concept})
    print("Response from OpenAI:", response)


def call_composed_chain():

    prompt = ChatPromptTemplate.from_template(
        "You are a python expert, explain me {concept}")
    chain = prompt | model | StrOutputParser()
    prompt_analyzer = ChatPromptTemplate.from_template(
        "Give me two line summary of {concept2}")

    composed_chain = {
        'concept2': chain} | prompt_analyzer | model | StrOutputParser()

    response = composed_chain.invoke({'concept': 'inheritance'})
    print("Response from Composed Chain:", response)


def add_document_to_pinecone():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=10)
    with open("Cystic-fibrosis.txt", "r") as file:
        doc_text = file.read()
    # chunk the document into smaller pieces
    document = text_splitter.create_documents([doc_text])
    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_keys())
    # create a pinecone index
    pc = Pinecone(api_key=get_pinecone_keys())
    index_name = "langchain-tutorial-index"
    if index_name not in pc.list_indexes().names():
        index = pc.create_index(name=index_name, dimension=1536, metric="cosine",
                                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    else:
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        uuids = [str(uuid4()) for _ in range(len(document))]
        vector_store.add_documents(documents=document, id=uuids)

    print("Document added to Pinecone index")


def pinecone_sample():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=10)
    # read the rett_text into a string
    with open("Rett syndrome.txt", "r") as file:
        rett_text = file.read()
    # chunk the document into smaller pieces
    document = text_splitter.create_documents([rett_text])
    # print(document)
    # convert the document into embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_keys())
    # embedded_document = embeddings.embed_documents(rett_text)
    # create a pinecone index
    pc = Pinecone(api_key=get_pinecone_keys())
    index_name = "langchain-tutorial-index"
    if index_name not in pc.list_indexes().names():
        index = pc.create_index(name=index_name, dimension=1536, metric="cosine",
                                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        uuids = [str(uuid4()) for _ in range(len(document))]
        vector_store.add_documents(documents=document, id=uuids)

    # connect to a pinecone index
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    results = vector_store.similarity_search(
        "What is Rett syndrome?",
        k=2
    )
    query_results = []
    for res in results:
        query_results.append(res.page_content)
    # flatten the list
    query_result_str = ' '.join(query_results)
    messages = [
        HumanMessage(content=query_result_str),
        SystemMessage(
            content="You only summarize the message, don't add any extra information."),
    ]
    response = model.invoke(messages)
    print("\n\033[92mResponse from Just Document:\n\033[0m")
    print(response.content)

    messages = [
        HumanMessage(content=query_result_str),
        SystemMessage(
            content="You are an expert in rare diseas, add your extra comments to the information."),
    ]
    response = model.invoke(messages)
    print("\n\033[91mResponse from OpenAI Expert:\n\033[0m")
    print(response.content)


def search_pincone():
    index_name = "langchain-tutorial-index"
    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_keys())
    pc = Pinecone(api_key=get_pinecone_keys())
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    query = input("Enter your query: ")
    results = vector_store.similarity_search(
        query,
        k=2
    )
    query_results = []
    for res in results:
        # print(f"* {res.page_content} [{res.metadata}]")
        query_results.append(res.page_content)

    # flatten the list
    query_result_str = ' '.join(query_results)
    messages = [
        HumanMessage(content=query_result_str),
        SystemMessage(
            content="You only summarize the message, don't add any extra information."),
    ]
    response = model.invoke(messages)
    print("\n\033[92mResponse from Document:\n\033[0m")
    print(response.content)

    messages = [
        HumanMessage(content=query_result_str),
        SystemMessage(
            content="You are an expert in rare diseas, add your extra comments to the information."),
    ]
    response = model.invoke(messages)
    print("\n\033[91mResponse from OpenAI Expert:\n\033[0m")
    print(response.content)


if __name__ == '__main__':
    call_llm()
    # call_with_prompt()
    # call_chain()
    # call_composed_chain()
    # pinecone_sample()
    # add_document_to_pinecone()
    # search_pincone()
