from markdownfile_load import load_markdown_files

# Load all markdown files from a folder
docs = load_markdown_files("data")

import os
import pandas as pd
from markdownfile_load import load_markdown_files
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


from langchain.prompts import ChatPromptTemplate


groq_key = "Need Groq API for LLM"
api_key = os.environ["GROQ_API_KEY"] = groq_key
secret_value_0="Hugging face API"

docs = load_markdown_files("data")
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
doc_chunks = text_splitter.create_documents(docs)
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

embedded_chunks = [embeddings.embed_query(chunk.page_content) for chunk in doc_chunks]
import chromadb

chroma_client = chromadb.PersistentClient(path="chromadb_store")
try:
    collection = chroma_client.create_collection("ubuntu_docs")
except:
    collection = chroma_client.get_collection("ubuntu_docs")

for i, chunk in enumerate(doc_chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk.page_content],
        embeddings=[embedded_chunks[i]],
    )
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize ChromaDB client and load the collection
chroma_client = chromadb.PersistentClient(path="chromadb_store")
collection = chroma_client.get_collection("ubuntu_docs")

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrieve_docs(query, top_k=3):
    """
    Retrieves the top K most relevant documents from ChromaDB.
    """
    query_embedding = embedding_model.embed_query(query)

    # Search for similar documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract document texts
    retrieved_docs = results["documents"][0] if results["documents"] else []
    return retrieved_docs


# template = """You are an assistant for question-answering tasks.
# Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know.
# Use three sentences maximum and keep the answer concise.
# if context not there please use your own context and give answer, but do not inform user that you dont have context
# Question: {question}
# Context: {context}
# Answer:
#
# """
# prompt = ChatPromptTemplate.from_template(template)
#
#
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0.20,
# )
#
# rag_chain = (
#     {
#         "context": RunnablePassthrough(),  # This just passes the context along
#         "question": RunnablePassthrough()  # This passes the question along
#     }
#     | prompt   # Combine the context and question into the prompt template
#     | llm      # The LLM processes the generated prompt
#     | StrOutputParser()  # Parse the output into a string
# )
#
# query = "How do I install packages in Ubuntu?"
#
# retrieved_documents = retrieve_docs(query)
#
# context = retrieved_documents
# response = rag_chain.invoke({"context": context, "question": query})
#
# # Print the result
# print(response)
#
