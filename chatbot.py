from langchain_openai import ChatOpenAI
from prompts import cv_assistant_prompt_template
from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import chainlit as cl
# from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

LLM_API_SERVER = os.getenv('LLM_API_SERVER')

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')

DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH', 'content')

pdf_loader = PyPDFDirectoryLoader(path=DOCUMENTS_PATH)
documents = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100)
documents_chunks = text_splitter.split_documents(documents)
EMBEDDING_MODEL_NAME = 'model'
print(EMBEDDING_MODEL_NAME)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

store = LocalFileStore('./.cache/')

# cached_embedder = CacheBackedEmbeddings.from_bytes_store(
#     embeddings, store, namespace=EMBEDDING_MODEL_NAME
# )
# db = FAISS.from_documents(
#     documents_chunks, cached_embedder)


db = Chroma.from_documents(documents_chunks, embeddings, client_settings = Settings(anonymized_telemetry=False))



@cl.on_chat_start
async def query_llm():
    await cl.Message(
        content=f"Guten Tag! Ich freue mich, dass Sie mit mir über mich chatten möchten. Bitte stellen Sie mir Ihre Fragen.",
    ).send()

    llm = ChatOpenAI(temperature=0,
                     # local llama.cpp-server,
                     openai_api_base=LLM_API_SERVER,
                     openai_api_key='not_needed',
                     )

    conversation_memory = ConversationBufferMemory(memory_key='chat_history',
                                                   max_len=50,
                                                   return_messages=True,
                                                   )

# CHAIN PORTION
    llm_chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=db.as_retriever(),
                                            chain_type_kwargs={
                                                'verbose': True,
                                                'prompt': cv_assistant_prompt_template,
                                                'memory': ConversationBufferMemory(
                                                    memory_key='history',
                                                    input_key='question')
                                            }
                                            )

    cl.user_session.set('llm_chain', llm_chain)
    cl.user_session.set('db', db)


@cl.on_message
async def query_llm(message: cl.Message):

    llm_chain = cl.user_session.get('llm_chain')

    response = await llm_chain.acall(message.content,
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])

    await cl.Message(
        content=response['result'].replace('<|eot_id|>', ''), author='PAItrick'
    ).send()

