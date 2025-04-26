# download_models.py
# Run this script once while you have internet connection to cache all required models

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch

print("Starting model download process...")

# Create directories for saving models
os.makedirs("models/llm", exist_ok=True)
os.makedirs("models/embeddings", exist_ok=True)

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Download and save LLM
print(f"Downloading LLM: {LLM_MODEL_ID}")
try:
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    tokenizer.save_pretrained("models/llm")
    
    # Download model
    print("Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    model.save_pretrained("models/llm")
    print("LLM downloaded and saved successfully!")
except Exception as e:
    print(f"Error downloading LLM: {e}")

# Download and save embedding model
print(f"\nDownloading embedding model: {EMBEDDING_MODEL_ID}")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)
    embedding_model.save("models/embeddings")
    print("Embedding model downloaded and saved successfully!")
except Exception as e:
    print(f"Error downloading embedding model: {e}")

print("\nDownload process complete! You can now use the models offline.")