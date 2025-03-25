from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from rag_code import retrieve_docs


def rag_chain(question):
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    if context not there please use your own context and give answer, but do not inform user that you dont have context
    Question: {question}
    Context: {context}
    Answer:
    
    """
    prompt = ChatPromptTemplate.from_template(template)


    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.20,
    )

    rag_chain = (
        {
            "context": RunnablePassthrough(),  # This just passes the context along
            "question": RunnablePassthrough()  # This passes the question along
        }
        | prompt   # Combine the context and question into the prompt template
        | llm      # The LLM processes the generated prompt
        | StrOutputParser()  # Parse the output into a string
    )

    query = question

    retrieved_documents = retrieve_docs(query)

    context = retrieved_documents
    response = rag_chain.invoke({"context": context, "question": query})
    return response

question="How to install Ubuntu"
answer=rag_chain(question)

