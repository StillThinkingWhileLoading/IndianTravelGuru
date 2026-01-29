# IndianTravelGuru
<img width="2384" height="1668" alt="image" src="https://github.com/user-attachments/assets/d1263872-8533-4055-954f-5a0757339da0" />

Overview

India Travel Guru is an intelligent travel planning assistant designed to help users create personalized itineraries for trips within India. Powered by AI, it leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware recommendations based on travel data from PDF documents.

This bot uses:

Llama 3.1:8B (via Ollama) as the core language model for generating responses.
FAISS for efficient vector search and retrieval.
Hugging Face Embeddings (all-MiniLM-L6-v2) for document embedding.
Streamlit for a user-friendly web interface.
Unstructured for parsing PDF travel data.
Whether you're planning a family vacation to Rajasthan or a solo adventure in Kerala, the bot suggests itineraries, budgets, activities, and more, tailored to your query.

Key Features

Personalized Itineraries: Generate trip plans based on duration, destination, group size, budget, and timing.
RAG-Powered Insights: Retrieves relevant information from a knowledge base of travel PDFs (e.g., guides, brochures).
Conversational Interface: Maintains chat history for seamless follow-up questions.
Easy Deployment: Runs locally or on Streamlit Cloud.
Extensible: Add more PDFs to the ./travel_data folder to expand the knowledge base.
Architecture

The application follows a modular RAG architecture:

Data Ingestion:
PDFs from the ./travel_data directory are loaded and parsed using unstructured.partition.pdf.
Documents are split into chunks (1000 characters with 200 overlap) using RecursiveCharacterTextSplitter.
Vector Store:
Embeddings are generated with Hugging Face's all-MiniLM-L6-v2 model.
Chunks are indexed in FAISS and saved locally at ./faiss_travel_index for fast similarity search.
If the index exists, it's loaded; otherwise, it's created on first run.
Retrieval:
User queries are passed to the FAISS retriever (top-5 relevant chunks).
Retrieved context is injected into the prompt.
Generation:
Ollama's Llama 3.1:8B model generates responses.
Prompt includes context, chat history, current date, and assumed user location (India).
The chain is built using LangChain's Runnable components for efficient processing.
User Interface:
Streamlit provides a chat-based UI with message history and a spinner for loading states.
Initial greeting guides users on query format for optimal results.
High-Level Flow

User Query → Retriever (FAISS) → Context Fetch → Prompt Assembly → LLM (Llama 3.1) → Response → Streamlit Display
Requirements

Hardware

CPU/GPU: Runs on standard hardware; GPU recommended for faster embeddings and inference.
RAM: At least 8GB (more for larger PDF datasets).
Disk Space: ~5GB for models and indexes (Llama 3.1:8B is ~4.7GB).
Software

Python 3.10+ (tested on 3.12).
Ollama: For running Llama 3.1 locally (install from ollama.com).
NLTK: For text processing (downloaded automatically).
Python Dependencies

Install via pip (see requirements.txt below for a sample):

langchain
langchain-community
langchain-huggingface
langchain-ollama
langchain-core
streamlit
unstructured[pdf]  # Includes pdf dependencies
faiss-cpu  # Or faiss-gpu if you have CUDA
huggingface-hub
nltk
torch  # For Hugging Face models
Create a requirements.txt file with the above and run:

pip install -r requirements.txt
Models

Embeddings: all-MiniLM-L6-v2 (auto-downloaded via Hugging Face).
LLM: Pull Llama 3.1:8B with Ollama:
ollama pull llama3.1:8b
Data Preparation

Place travel-related PDFs (e.g., destination guides, hotel lists) in ./travel_data/.
Ensure PDFs are text-extractable; scanned images may require OCR (not handled natively).
Installation

Clone the repository:
git clone https://github.com/yourusername/india-travel-guru-bot.git
cd india-travel-guru-bot
Install dependencies:
pip install -r requirements.txt
Install Ollama and pull the model:
Download and install Ollama from ollama.com.
Run: ollama pull llama3.1:8b
Prepare data:
Create a ./travel_data folder and add your PDF files.
The FAISS index will be auto-generated on first run.
How to Run the Application

Start Ollama server (if not running):
ollama serve
Run the Streamlit app:
streamlit run app.py  # Assuming the code is in app.py
Access the app:
Open your browser at http://localhost:8501.
Enter a query like: "Plan a 10-day trip to Rajasthan for a family of 4 in December 2025 with moderate budget."
Chat with the bot for refinements!
Debugging Tips

If PDFs fail to load: Check for corrupted files or install missing system deps (e.g., poppler for PDF parsing).
Index not found? Delete ./faiss_travel_index and restart to rebuild.
Slow performance? Increase num_ctx in Ollama or use a smaller model.
Errors with deserialization? Ensure allow_dangerous_deserialization=True is set (as in code).
Deployment

Streamlit Cloud: Push to GitHub and deploy via share.streamlit.io. Note: Ollama may require custom setup (use cloud-hosted LLMs if needed).
Docker: Create a Dockerfile for containerization (coming soon!).
Heroku/Vercel: Possible, but Ollama integration may need alternatives like Groq or OpenAI.
Contributing

Contributions welcome! Fork the repo, create a branch, and submit a PR. Focus on:

Adding more travel data sources.
Improving prompt engineering.
Enhancing UI with maps or images.
License

MIT License. See LICENSE for details.

Acknowledgments

Built with ❤️ using open-source tools from LangChain, Hugging Face, Ollama, and Streamlit.
Inspired by India's rich travel destinations!
If you enjoy this project, star the repo and share your travel stories! 🌟
