"""
[ V ] Cloudflare 공식문서를 위한 SiteGPT 버전을 만드세요.
[ V ]챗봇은 아래 프로덕트의 문서에 대한 질문에 답변할 수 있어야 합니다:
    AI Gateway
    Cloudflare Vectorize
    Workers AI
[ V ] 사이트맵을 사용하여 각 제품에 대한 공식문서를 찾아보세요.
[ V ] 여러분이 제출한 내용은 다음 질문으로 테스트됩니다:
    "llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?"
    "Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?"
    "벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?"
[ V ] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
[ V ] st.sidebar를 사용하여 Streamlit app과 함께 깃허브 리포지토리에 링크를 넣습니다.
"""

import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import asyncio
import nest_asyncio


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
)

st.title("SiteGPT")

st.markdown("""
Ask questions about the content of a website.

Start by writing the URL of the website on the sidebar.
""")


with st.sidebar:
    key = st.text_input("Input your OpenAI API Key!")
    url = st.text_input("Write down a URL", placeholder="Https://example.com")
    st.write("Github: https://github.com/oliv3h/FULLSTACK-GPT-CHALLENGE/blob/main/pages/03_SITE.py")


nest_asyncio.apply()

def parse_page(soup):
    # SitemapLoader에서는 SOAP 사용
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")

@st.cache_data(show_spinner="Searching Website...")
def load_website(url):

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )

    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        # filter_urls=["https://openai.com/blog/data-partnerships"]) or r"^(.*\/blog\/).*" --blog 포함 or r"^(?!.*\/blog\/).* --blog 제외"
        filter_urls=[r"^(.*\/workers-ai\/).*", r"^(.*\/ai-gateway\/).*", r"^(.*\/vectorize\/).*"]
    )
    loader.requests_per_second = 5 # 너무 빠게 설정하면 (기본=1초) 차단될 수 있음

    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=key))
    return vector_store.as_retriever() 

if key:
    llm = ChatOpenAI(
        temperature=0.1,
        api_key=key
    )

answers_prompt = ChatPromptTemplate.from_template(
    """
Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

Then, giva a score to the answer between 0 and 5. 0 being not helpful to the user
and 5 being helpful to the user.

Examples:                                            

Question: How for away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know.
Score: 0

Your turn!

Context: {context}
Question: {question}
""")

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain =  answers_prompt | llm
    
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #        "question": question,
    #        "context": doc.page_content
    #    })
    #    answers.append(result)
    return {"question": question, "answers":[
        {
            "answer": answers_chain.invoke({"question": question, "context": doc.page_content}).content,
            "source": doc.metadata["source"],
            "date": doc.metadata["lastmod"]
        } for doc in docs
    ]}

choose_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    Use ONLY the following pre-existing answers to answer the user's question.
    Use the answers that have the highest score (more helpful) and favor the most recent ones.

    Return the source of the answers as they are, do not change them.

    Answers: {answers}
     """),
     ("human", "{question}")
])

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condenced = "\n\n".join(f"Answer:{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers)
    return choose_chain.invoke({
        "question": question,
        "answers": condenced
    })

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website!")
        if query:
            chain = {"docs": retriever, "question": RunnablePassthrough()} | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
