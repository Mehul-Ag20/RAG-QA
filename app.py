# app_offline.py

import os
import torch
import gradio as gr
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.chains import RetrievalQA
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# Set environment variable for offline use
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/llm"  # Local path to saved model
EMBEDDING_MODEL_PATH = "models/embeddings"  # Local path to saved embedding model
DB_DIRECTORY = "vectordb"
DOCUMENTS_DIRECTORY = "files"

def initialize_model():
    """Initialize the LLM with quantization for limited VRAM from local files."""
    print("Loading model from local files...")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and tokenizer from local paths
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    
    # Create text generation pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=False
    )
    
    # Create LangChain LLM
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    print("Model loaded successfully!")
    return llm

def load_documents():
    """Load documents from the files directory."""
    print(f"Loading documents from {DOCUMENTS_DIRECTORY}...")
    
    # Check if directory exists
    if not os.path.exists(DOCUMENTS_DIRECTORY):
        os.makedirs(DOCUMENTS_DIRECTORY)
        print(f"Created directory {DOCUMENTS_DIRECTORY} as it did not exist")
        return []
    
    # Configure loaders for different file types
    loaders = {
        ".pdf": DirectoryLoader(DOCUMENTS_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader),
        ".txt": DirectoryLoader(DOCUMENTS_DIRECTORY, glob="**/*.txt", loader_cls=TextLoader),
        ".docx": DirectoryLoader(DOCUMENTS_DIRECTORY, glob="**/*.docx", loader_cls=Docx2txtLoader),
    }
    
    # Load all documents
    documents = []
    for file_type, loader in loaders.items():
        try:
            documents.extend(loader.load())
            print(f"Loaded {file_type} documents")
        except Exception as e:
            print(f"Error loading {file_type} documents: {e}")
    
    print(f"Loaded {len(documents)} documents in total")
    return documents

def create_vector_db(documents):
    """Process the documents and create a vector database."""
    print("Creating vector database...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks")
    
    # Initialize embeddings from local files
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create or load the vector store
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIRECTORY
    )
    db.persist()
    
    print("Vector database created and persisted successfully!")
    return db

def create_qa_chain(llm, db):
    """Create a retrieval QA chain."""
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    
    return qa_chain

def format_response(result):
    """Format the response for display."""
    answer = result["result"]
    source_docs = result.get("source_documents", [])
    
    formatted_sources = []
    for i, doc in enumerate(source_docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted_sources.append(f"Source {i+1}: {source}, Page: {page}")
    
    if formatted_sources:
        source_text = "\n\nSources:\n" + "\n".join(formatted_sources)
    else:
        source_text = "\n\nNo sources found"
    
    return answer + source_text

def query_documents(question):
    """Process a query and return the answer."""
    if not question.strip():
        return "Please enter a question."
    
    try:
        result = qa_chain({"query": question})
        return format_response(result)
    except Exception as e:
        return f"Error: {str(e)}"

def validate_offline_setup():
    """Validate that the necessary offline files are available."""
    missing_components = []
    
    if not os.path.exists(MODEL_PATH):
        missing_components.append(f"LLM model at {MODEL_PATH}")
    
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        missing_components.append(f"Embedding model at {EMBEDDING_MODEL_PATH}")
        
    if missing_components:
        print("ERROR: Missing required offline components:")
        for component in missing_components:
            print(f"- {component}")
        print("\nPlease run the download_models.py script while connected to the internet first.")
        return False
        
    return True

# Initialize everything
print("Initializing the offline RAG system...")

# First validate we have all needed offline files
if not validate_offline_setup():
    print("Exiting due to missing offline components.")
    exit(1)

# Load documents and initialize model
documents = load_documents()
llm = initialize_model()
db = create_vector_db(documents)
qa_chain = create_qa_chain(llm, db)
print("RAG system initialized successfully!")

# Create Gradio interface
demo = gr.Interface(
    fn=query_documents,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=gr.Textbox(lines=10),
    title="Offline RAG-based Q&A System with Mistral 7B",
    description="Ask questions about your documents and get answers with source references.",
)

if __name__ == "__main__":
    demo.launch()