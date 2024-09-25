import os
import chainlit as cl
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Initialize global variables
embeddings = None
vectors = None
agent = None

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

@cl.on_chat_start
async def start():
    # Prompt the user to upload a PDF file
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to begin.",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=180,
    ).send()

    if not files:
        await cl.Message(content="No file was uploaded. Please refresh and try again.").send()
        return

    file = files[0]

    # Process the uploaded PDF
    await cl.Message(content="Processing and embedding the PDF...").send()

    global embeddings, vectors, agent

    # Initialize embeddings if not already done
    if embeddings is None:
        embeddings = NVIDIAEmbeddings()

    # Load the PDF using PyPDFLoader
    pdf_loader = PyPDFLoader(file.path)
    docs = pdf_loader.load()

    # Create chunks from the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.split_documents(docs)

    # Create vector embeddings for the documents
    vectors = FAISS.from_documents(final_documents, embeddings)

    # Initialize the LLM
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

    # Create the retrieval tool
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Question: {input}
        """
    )))
    
    retrieval_tool = Tool(
        name="Document_QA",
        func=retrieval_chain.invoke,
        description="Useful for answering questions about the uploaded document."
    )

    # Initialize the agent with tools
    tools = [retrieval_tool, calculator]
    agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True)

    # Initialize an empty chat history
    cl.user_session.set("chat_history", [])

    await cl.Message(content="PDF processed and embedded. You can now ask questions about the document or use other tools.").send()

@cl.on_message
async def main(message: cl.Message):
    global agent

    if agent is None:
        await cl.Message(content="Please upload a PDF file before asking questions.").send()
        return

    # Get the chat history
    chat_history = cl.user_session.get("chat_history", [])

    # Convert chat history to the format expected by the agent
    formatted_history = []
    for human, ai in chat_history:
        formatted_history.append({"role": "user", "content": human})
        formatted_history.append({"role": "assistant", "content": ai})

    # Use the agent to process the user's question
    response = await cl.make_async(agent.run)(
        input=message.content,
        chat_history=formatted_history
    )

    # Update the chat history
    chat_history.append((message.content, response))
    cl.user_session.set("chat_history", chat_history)

    await cl.Message(content=response).send()

if __name__ == "__main__":
    cl.run()