from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
import dotenv
dotenv.load_dotenv()

groq_api = os.getenv('GROQ_API_KEY')
model=ChatGroq(model='Gemma2-9b-It', groq_api_key=groq_api)

# Chat prompt
generic_prompt = "translate the following into {lang} Language:"

prompt = ChatPromptTemplate.from_messages(
    [('system',generic_prompt), ('user',"{user_input}")]
    )

parser = StrOutputParser()

# Chain

chain = prompt|model|parser

# ### APP Definition
# app = FastAPI(title="Langchain Server",
#               version=1.0,
#               description="simple api server using runnable interface")

# add_routes(
#     app,
#     chain,
#     path='/chain'
# )

# if __name__ =="__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)