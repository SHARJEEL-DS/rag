# Save this as app.py
import streamlit as st
import subprocess
from transformers import BitsAndBytesConfig
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.prompts.prompts import SimpleInputPrompt
import torch
# Initialize Streamlit app
st.title("Llama2 Q&A System")
st.write("Ask questions, and the Llama2 model will respond.")

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

# Set up system prompt
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided.
If you don't know the answer, just say that you don't know.
"""
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# Configure model
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    model_kwargs={  "torch_dtype": torch.float16,  "quantization_config": quantization_config}
)

# Initialize embeddings and index
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)

# Take user input
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if query:
        # Query the model
        response = query_engine.query(query)
        st.write("**Response:**")
        st.write(response)
    else:
        st.write("Please enter a question.")
