from ollama import chat
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embedding= HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

st.title('RAG chatbot with message history')
session_id = st.text_input('Enter Session ID:', value='default_session')
llm = ChatGroq(model='openai/gpt-oss-20b',
               groq_api_key=os.getenv('GROQ_API_KEY'))
if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files=st.file_uploader('Choose your pdf', type='pdf',accept_multiple_files=True)

if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf='./temp.pdf'
        with open(temppdf, 'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

        loaded_pdf = PyPDFLoader(temppdf).load()
        documents.extend(loaded_pdf)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(documents=docs, embedding=embedding)
    retriever = vector_store.as_retriever()

    q_prompt = (
        "Given the chat history and the user's latest question,"
        "formulate a standalone meaningful question that can be furthur answered"
        "do not try to related it forcefully only relate it if its related and then create the standalone question"
    )

    q_prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', q_prompt),
            MessagesPlaceholder('chat_history'),
            ("human", "{input}")
        ]
    )
    history_aware_retriever = create_history_aware_retriever(retriever, llm, q_prompt_template)

    prompt = (
        "You are a very helpful assistant for question answering"
        "Use the following retrieved context to answer the question"
        "keep the answer short unless user mentions to give the answer in detail"
        "\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', prompt),
            MessagesPlaceholder('chat_history'),
            ("human", "{input}")
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, prompt_template, document_variable_name="context")
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def get_session_history(session):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conv_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    user_input = st.text_input("Your Question: ")
    if user_input:
        session_history = get_session_history(session_id)
        response = conv_rag_chain.invoke(
            {'input': user_input},
            config={
                "configurable" : {"session_id":session_id}
            }
        )
        st.write(response)