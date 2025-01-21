"""
[ V ] 함수 호출을 사용합니다.
[ V ] 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
[ V ] 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
[ V ] 만점이면 st.ballons를 사용합니다.
[ V ] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
[ V ] st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.
"""

import json
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json","")
        return json.loads(text)
    
output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="?"
)

st.title("QuizGPT")

@st.cache_data(show_spinner="Loading File...")
def split_file(file):
    file_content = file.read()
    file_path = f'./.cache/quiz_files/{file.name}'
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    # Hash할 수 없는 매개변수가 있거나, Streamlit이 데이터의 서명을 만들 수 없는 경우, 위와 같이 매개변수를 처리
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Making Quiz...")
def run_function_call_chain(_docs, level, topic):
    chain = use_function_calling_prompt | llm
    response = chain.invoke({"context": _docs, "level": level})
    response = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(response)

@st.cache_data(show_spinner="Searching Wiki..")
def wiki_search(_term):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(_term)

with st.sidebar:
    key = st.text_input("OpenAI API Key")
    docs = None
    level = st.selectbox("Choose The Level of Difficult.", (
        "Easy", "Hard"
    ))
    choice = st.selectbox("Choose What you want to use", (
        "File", "Wikipedia Article",
    ),)
    st.write("Github: https://github.com/oliv3h/FULLSTACK-GPT-CHALLENGE/blob/main/pages/02_QUIZ.py")
    if choice == "File":
        file = st.file_uploader("Upload a .docs, .txt or .pdf file", type=["pdf", "txt", "docs"],)
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if key:
    function = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    }

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        api_key=key
    ).bind(
        function_call={
            "name": "create_quiz"
        },
        functions=[function]
    )


    questions_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                """
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 questions to test the user's
    knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and 
    one should be correct.

    Use (o) to signal the correct answer.

    Question examples:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital of Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avater released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caeser?
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    Your turn!

    Context: {context}
                """)
            ]
        )

    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    question_chain = {"context": format_docs} | questions_prompt | llm

    questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
        Use (o) to signal the correct answer.
            
        Question examples:
            
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
            
        Your turn!
            
        Context: {context}
    """,
            )
        ]
    )
    questions_chain = {"context": format_docs} | questions_prompt | llm
    formatting_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a powerful formatting 
        You format exam questions into JSON format.
        Answers with (o) are the correct ones.
        
        Example Input:
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
        
        
        Example Output:
        
        ```json
        {{ "questions": [
                {{
                    "question": "What is the color of the ocean?",
                    "answers": [
                            {{
                                "answer": "Red",
                                "correct": false
                            }},
                            {{
                                "answer": "Yellow",
                                "correct": false
                            }},
                            {{
                                "answer": "Green",
                                "correct": false
                            }},
                            {{
                                "answer": "Blue",
                                "correct": true
                            }},
                    ]
                }},
                            {{
                    "question": "What is the capital or Georgia?",
                    "answers": [
                            {{
                                "answer": "Baku",
                                "correct": false
                            }},
                            {{
                                "answer": "Tbilisi",
                                "correct": true
                            }},
                            {{
                                "answer": "Manila",
                                "correct": false
                            }},
                            {{
                                "answer": "Beirut",
                                "correct": false
                            }},
                    ]
                }},
                            {{
                    "question": "When was Avatar released?",
                    "answers": [
                            {{
                                "answer": "2007",
                                "correct": false
                            }},
                            {{
                                "answer": "2001",
                                "correct": false
                            }},
                            {{
                                "answer": "2009",
                                "correct": true
                            }},
                            {{
                                "answer": "1998",
                                "correct": false
                            }},
                    ]
                }},
                {{
                    "question": "Who was Julius Caesar?",
                    "answers": [
                            {{
                                "answer": "A Roman Emperor",
                                "correct": true
                            }},
                            {{
                                "answer": "Painter",
                                "correct": false
                            }},
                            {{
                                "answer": "Actor",
                                "correct": false
                            }},
                            {{
                                "answer": "Model",
                                "correct": false
                            }},
                    ]
                }}
            ]
        }}
        ```
        Your turn!
        Questions: {context}
    """,
            )
        ]
    )

    formatting_chain = formatting_prompt | llm

    use_function_calling_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

        The questions has Level of Difficult. This Questions has {level} level.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.

        Context: {context}
    """,
            )
        ]
    )


if not docs:
    # 사용자를 반기는 글
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test
    your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_function_call_chain(docs, level, topic if topic else file.name)

    count = 0
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio("Select an option.", 
                     [answer["answer"] for answer in question["answers"]],
                     index=None)
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                count += 1
            elif value is not None:
                st.error("Wrong.")
        button = st.form_submit_button()
        if count == 10:
            st.balloons()
