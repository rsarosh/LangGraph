import configparser


def get_openai_keys():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['OpenAI_KEYS']


def get_tavily_api_keys():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['TAVILY_API_KEY']

def get_pinecone_keys():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['PINECONE_API_KEY']

def get_deepseek_keys():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['DEEPSEEK_API_KEY']

def get_anthropic_keys():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['ANTHROPIC_API_KEY']
