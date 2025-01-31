# Steps to Run the Python Project

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/LangGraph.git
    cd LangGraph
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
      ```sh
      venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```sh
      source venv/bin/activate
      ```

4. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

5. **Run the project:**
    Rename config copy.ini and fill the keys for: 

    OpenAI

    PineCone

    Tavily

    See below for the URL links to go their portal and obtain keys
    ```sh
    python __init__.py
    ```

6. **Deactivate the virtual environment (optional):**
    ```sh
    deactivate
    ```

# Links
## Pinecone
https://docs.pinecone.io/guides/get-started/overview

## OpenAI
https://platform.openai.com/docs/overview

## Tavily

https://app.tavily.com/home?code=htf-dx2MTxCv2xWBJJDO38De2tyo2VbpCuRgruQ413U9n&state=eyJyZXR1cm5UbyI6Ii9ob21lIn0

## Langgraph
https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot


## Learn More
https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot

# For human in loop
```pip3 install langgraph-cli```

```pip install lagngraph-sdk```


complete the langgraph.json file as per your project

human_in_loop_deployed.py is the file to test human in loop scenarios.

## To run 
Make sure docker is install. 

```langgraph dev```

to deploy basic chatbot, update the langgraph.json with this line
    "basic_chatbot": "./basic_chatbot_deployed.py:basic_chatbot_deployed"

