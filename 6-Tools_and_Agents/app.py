from platform import system_alias
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import AgentType, initialize_agent, create_openai_tools_agent
import streamlit as st
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

groq_api = os.getenv('GROQ_API_KEY')

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name='Search')


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api,
    streaming=True
)

system_message = """You are a helpful assistant.
- If the user asks something casual (greetings, jokes, small talk), just reply normally.
- Only call tools if the user’s query requires external knowledge.
"""

tools = [search, arxiv_tool, wiki]

search_agent = initialize_agent(
    tools=[search, arxiv_tool, wiki],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    early_stopping_method="generate",
    agent_kwargs={"system_message": system_message}
)

st.title('Langchain - Chat with Search')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role' : 'assistant', 'content' : 'Hi, i am an assistant chatbot who can also search the web!'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input():
    st.session_state.messages.append({'role' : 'user', 'content' : prompt})
    st.chat_message('user').write(prompt)

    
    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant', 'content': response})
        st.write(response)