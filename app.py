"""
이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
파일 업로드 및 채팅 기록을 구현합니다.
사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
"""
import streamlit as st
import time
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Ask Anything you want!", page_icon="📑")

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    message_box = st.empty()

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token:str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

key = ""

# st.session_state 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_data(show_spinner="Embedding File...")
def embed_file(file):
    file_content = file.read()
    file_path = f'./.cache/files/{file.name}'
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# 메세지를 세션에 저장하는 메서드
def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

# 메세지를 보냈을 때 표시하는 메서드
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 화면에 대화를 표시하는 메서드 (*streamlit는 무조건 같은 파일이 실행되므로 이와 같이 표시)
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# LLM에서 사용할 템플릿 선언
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", 
     """
    Answer the question using ONLY the following context. If you don't know the answer
    just say you don't know. DON'T make anything up.

    Context: {context}
     """),
    ("human", "{question}")
])

def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


st.markdown("""
Welcome!
            
Use ths chatbot to ask question to an AI about your files!
            
Upload your file at the Sidebar.
""")

with st.sidebar:
    key = st.text_input("OpenAI API Key")
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docs"])

if file:
    retriever = embed_file(file)
    send_message("I'm Ready! Ask Away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "Human")

        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
            api_key=key
        )       

        memory = ConversationBufferMemory(
            llm=llm,
            return_messages=True,
            memory_key="chat_history"
        )

        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
            memory.save_context(
                {"input": message},  # 입력값
                {"output": response.content}  # 출력값
            )
        # send_message(response.content, "ai") 
else:
    # 파일이 없을 때 초기화
    st.session_state["messages"] = []
