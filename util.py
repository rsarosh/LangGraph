import configparser
from IPython.display import Image


def get_openai_keys():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["DEFAULT"]["OpenAI_KEYS"]


def get_tavily_api_keys():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["DEFAULT"]["TAVILY_API_KEY"]


def get_pinecone_keys():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["DEFAULT"]["PINECONE_API_KEY"]


def get_deepseek_keys():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["DEFAULT"]["DEEPSEEK_API_KEY"]


def get_anthropic_keys():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["DEFAULT"]["ANTHROPIC_API_KEY"]


def save_image(image, filename):
    image_path = filename
    with open(image_path, "wb") as f:
        f.write(image.data)


def create_graph_image(graph):
    image = Image(graph.get_graph().draw_mermaid_png())
    return image


def create_and_save_gaph_image(graph, filename="graph_image.png"):
    image = create_graph_image(graph)
    save_image(image, filename)
