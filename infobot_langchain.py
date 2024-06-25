from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

def load_documents():
    # Load documents from the "data" directory
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_index(documents):
    # Use OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create a FAISS vector store index
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def get_query_engine():
    # Load the environment variables
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Load documents and create index
    documents = load_documents()
    vectorstore = create_index(documents)
    
    # Create and return a query engine from the index
    return vectorstore

def get_response(query):
    vectorstore = get_query_engine()
    llm = OpenAI()
    
    # Perform the query using the vector store and LLM
    docs = vectorstore.similarity_search(query)
    context = " ".join([doc.page_content for doc in docs])
    llm = OpenAI()
    response = llm(context + "\n\n" + query)
    return response