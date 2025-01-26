"""
[ ] 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
[ ] 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
[ V ] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
[ V ] st.sidebar를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.
"""

import streamlit as st
import openai as client
import datetime 
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from openai import OpenAI

st.set_page_config(
    page_title="OpenAI Assistants",
    page_icon="🤖"
)

st.markdown(
    """
    # OpenAI Assistants

    - 궁금한 것이 있다면 물어 보세요 :)
    """
)

with st.sidebar:
    api_key = st.text_input("Insert your API Key:")
    st.write("Github: https://github.com/oliv3h/FULLSTACK-GPT-CHALLENGE/blob/main/pages/05_ASSISTANT.py")

def createDownloadButton(content):
    st.download_button(
        label="Download",
        data = content.encode("utf-8"),
        file_name="Download_file.txt",
        mime="text/plain",
        key=datetime.datetime.now()
    )

def DuckDuckGoSearchTool(inputs):
    query = inputs["query"]
    search = DuckDuckGoSearchResults()
    return search.run(query)

def WikipediaSearchTool(inputs):
    query = inputs["query"]
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.run(query)

def WebScrapingTool(inputs):
    url = inputs["url"]
    loader = WebBaseLoader([url])
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    return text

def SaveToTXTTool(inputs):
    text = inputs["text"]
    with open("research_results.txt", "w") as file:
        file.write(text)
    return "Research results saved to research_results.txt"

function_maps = {
    "DuckDuckGoSearchTool": DuckDuckGoSearchTool,
    "WikipediaSearchTool": WikipediaSearchTool,
    "WebScrapingTool": WebScrapingTool,
    "SaveToTXTTool": SaveToTXTTool
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "DuckDuckGoSearchTool",
            "description": "Use this tool to perform web searches using the DuckDuckgo search engine. It takes a query as an argument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for"
                    }
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "WikipediaSearchTool",
            "description": "Use this tool to perform web searches using the Wikipedia. It takes a query as an argument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for"
                    }
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "WebScrapingTool",
            "description": "If you found the website link in DuckDuckGo, Use this to get the content of the link for my research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website you want to scrape"
                    }
                },
                "required": ["url"],
            }
        }
    }
]

def get_assistant_id():
    if "assistant_id" not in st.session_state:
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="""
            You are a research expert.

            Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 

            When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 

            Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.

            Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.

            The information from Wikipedia must be included.

            Ensure that the final .txt file contains detailed information, all relevant sources, and citations.
            """,
            tools=functions,
            model="gpt-4o-mini"
        )
        st.session_state["assistant_id"] = assistant.id
    else:
        return st.session_state["assistant_id"]

def get_thread_id():
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "assistant",
                    "content": "Please Let me know your question!",
                }
            ]
        )
        st.session_state["thread_id"] = thread.id
    return st.session_state["thread_id"]

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id
    )

def get_messages(thread_id):
    messages = list(
        client.beta.threads.messages.list(
            thread_id=thread_id,
        )
    )
    return list(reversed(messages))
    
def send_message(content):
    if "run_id" not in st.session_state:
        client.beta.threads.messages.create(
            thread_id=get_thread_id(),
            role="user",
            content=content
        )

        run = client.beta.threads.runs.create(
            thread_id=get_thread_id(),
            assistant_id=get_assistant_id()
        )

        st.session_state["run_id"] = run.id
    
    with st.status("Processing..."):
        while get_run(st.session_state["run_id"], get_thread_id()).status == "requires_action":
            submit_tool_outputs(st.session_state["run_id"], get_thread_id())

    last_message = get_messages(get_thread_id())[-1]
    if get_run(st.session_state["run_id"], get_thread_id()).status == "completed":
        with st.chat_message(last_message.role):
            st.markdown(last_message.content[0].text.value)

        createDownloadButton(last_message.content[0].text.value)
        print(last_message)
    elif get_run(st.session_state["run_id"], get_thread_id()).status == "failed":
        with st.chat_message("assistant"):
            st.markdown("Try Again later :()")

import json

def get_tool_output(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"calling function: {function.name} with arg {function.arguments}")
        outputs.append({
            "output":function_maps[function.name](json.loads(function.arguments)),
            "tool_call_id": action_id
        })
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_output(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs
    )


if api_key:
    query = st.chat_input("Ask a question everything you want!")
    client = OpenAI(api_key=api_key)
    
    assistant_id = get_assistant_id()

    for idx, message in enumerate(get_messages(get_thread_id())):
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value)
        if message.role == "assistant" and idx > 0:
            createDownloadButton(
                message.content[0].text.value
            )

            
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        send_message(query)
