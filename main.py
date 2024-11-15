# Save this as app.py
import streamlit as st
import subprocess
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# Initialize Streamlit app
st.title("GPT-2 Q&A System")
st.write("Ask questions, and the GPT-2 model will respond.")

# Hugging Face authentication
st.write("### Hugging Face Login")
token = st.text_input("Enter your Hugging Face API token", type="password")
if st.button("Login to Hugging Face"):
    if token:
        # Run huggingface-cli login with the provided token
        subprocess.run(["huggingface-cli", "login", "--token", token], check=True)
        st.success("Successfully logged in to Hugging Face!")
    else:
        st.error("Please enter a valid Hugging Face API token.")

# Load documents
st.write("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # Use a smaller model if needed for performance, e.g., "gpt2" or "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Initialize embeddings and index
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine()

# Take user input
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if query:
        # Encode the question with GPT-2
        inputs = tokenizer.encode(query, return_tensors="pt")
        outputs = model.generate(inputs, max_length=256, num_return_sequences=1)
        
        # Decode and display the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("**Response:**")
        st.write(response)
    else:
        st.write("Please enter a question.")
