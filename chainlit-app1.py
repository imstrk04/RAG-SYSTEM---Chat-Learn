import os
import chainlit as cl
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Initialize global variables
embeddings = None
vectors = None

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

    global embeddings, vectors

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

    await cl.Message(content="PDF processed and embedded. You can now ask questions about the document.").send()

@cl.on_message
async def main(message: cl.Message):
    global vectors

    if vectors is None:
        await cl.Message(content="Please upload a PDF file before asking questions.").send()
        return

    # Process the user's question
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = await cl.make_async(retrieval_chain.invoke)({'input': message.content})

    await cl.Message(content=response['answer']).send()

if __name__ == "__main__":
    cl.run()