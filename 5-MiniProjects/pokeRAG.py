import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def create_document_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.docs = PyPDFDirectoryLoader('PDFs').load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.chunks = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.chunks, st.session_state.embeddings)

st.title('Pokemon RAG Chatbot')

prompt = ChatPromptTemplate.from_template(
    """
    Answer to the given question based on context given to you.
    you your own knowledge and common sense whenever needed but do notify the user about it.
    <context>
    {context}
    </context>
    Question:{input}
    """
)

llm = ChatGroq(
    model='openai/gpt-oss-20b',
    groq_api_key=os.environ['GROQ_API_KEY']
)
btn =st.button('Load_documents')
if btn:
    create_document_embeddings()
    st.write("document is ready!")

user_input = st.text_input('Enter your Question:')

if user_input:
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    st.session_state.retriever = st.session_state.vectors.as_retriever()
    rag_chain = create_retrieval_chain(st.session_state.retriever, chain)

    response = rag_chain.invoke({"input": user_input})
    st.write(response['answer'])

