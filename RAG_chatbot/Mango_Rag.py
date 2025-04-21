import os
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from fastapi import FastAPI, Request
import uvicorn
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(
    title="MongoDB RAG",
    description="A simple API using FastAPI and RAG",
)

client = MongoClient(r"mongodb+srv://Ketanmahadik:Password%4029@cottondisease.dm1newg.mongodb.net/")

#* Define collection and index name
db_name = "cottondisease"
collection_name = "info"
atlas_collection = client[db_name][collection_name]
vector_search_index = "vector_index"

# Loading the environment variables
groq_api_key = os.getenv("groq_api_key")

# Defining the LLM
llm = ChatGroq(groq_api_key = groq_api_key, 
                model_name = 'allam-2-7b')

# Defining the Prompt
Prompt = PromptTemplate.from_template("""
You are a highly intelligent and professional assistant designed to answer user queries based strictly on the given context. Follow these instructions carefully:

1. Use **only the information provided in the context** to generate your response.
2. If the context does not contain sufficient information to answer the question, respond with a **polite and professional message** indicating that the information is not available.
3. Do not provide fabricated or assumed answers. Avoid speculation or generic filler content.
4. Maintain a **formal, clear, and concise tone** in every response. Avoid casual language, irrelevant details, or "thought processes."
5. Your response should **only include the final answer** — do not explain how you arrived at it, and do not repeat or reference the question.
6. Never mention the words "context" or "question" in your output.
7. IF the question is about diseses of cotton plants, then provide the answer the question from your own knowledge, else answer the question from the context.
8. if the question is about asking suggestion on prevention of diseases, then look for the database if not found then  provide the answer from your own knowledge.
 
<context>
{context}
</context>

User Question: {input}

""")

# Initializing retrieval chain and vector store
retrieval_chain = None
vector_store = None

# Function to load,split and embed the document
def load_embeddings():
    # Loading the context
    loader = PyPDFLoader(r"C:\Users\ketan\OneDrive\Desktop\cotton disease\RAG pdfs\Cotton Disease info pdf.pdf")
    docs = loader.load()

    # Splitting the loaded docs
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    split_docs = splitter.split_documents(docs)

    # Creating embeddings of the splitted data
    embeddings = OllamaEmbeddings(model='nomic-embed-text')

    # Creating a vector store form MongoDB Atlas
    global vector_store
    vector_store = MongoDBAtlasVectorSearch.from_documents(documents = split_docs,
                                                        embedding = embeddings,
                                                        collection = atlas_collection,
                                                        index_name = vector_search_index)

    # Updating the flag
    global loaded_embeddings
    loaded_embeddings = True

def load_vector_store():
    global vector_store

    # Creating the vector store from existing MongoDB collection
    embeddings = OllamaEmbeddings(model='nomic-embed-text') 
    vector_store = MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=atlas_collection,
        index_name=vector_search_index
    )

# Initializing a flag
# Check if embeddings already exist in MongoDB
if atlas_collection.count_documents({}) > 0:  
    load_vector_store()
    loaded_embeddings = True
    print("Embeddings already loaded in MongoDB.")
else:    
    load_embeddings()

# Retrieval Mechanism
# Creating a document chain
doc_chain = create_stuff_documents_chain(llm = llm, prompt = Prompt, output_parser= StrOutputParser())

# Defining the vector store as retriever
retriever = vector_store.as_retriever( search_type = "similarity",
search_kwargs = { "k": 10 })

# Creating a retrieval chain
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

# Define a route
@app.post("/api")
async def invoke_retrieval(request: Request):
    input_data = await request.json()
    query = input_data.get("input", {}).get("input")
    if query:
        response = retrieval_chain.invoke({"input": query})
        return {"response": response['answer']}
    return {"response": "No query provided"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")