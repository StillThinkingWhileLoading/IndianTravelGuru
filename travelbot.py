import os
from pathlib import Path
from datetime import date
import streamlit as st

# === PDF loading (unstructured) ===
from unstructured.partition.pdf import partition_pdf
import nltk

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def load_pdfs_from_directory(directory_path):
    all_documents = {}
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        print(f"Error: Directory not found → {directory_path}")
        return None

    for pdf_path in path.glob("*.pdf"):
        print(f"Processing: {pdf_path.name} ...")
        try:
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",
                infer_table_structure=True,
                model_name="yolox"
            )
            text = "\n\n".join(str(el) for el in elements)
            all_documents[pdf_path.name] = text
        except Exception as e:
            print(f"Failed {pdf_path.name}: {e}")

    return all_documents


# Change this to your actual folder
PDF_FOLDER = "./travel_data"  # or r"C:\Users\...\travel_data" on Windows

# === Create or load FAISS index ===
INDEX_PATH = "./faiss_travel_index"

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ────────────────────────────────────────────────
# Load or create vector store
# ────────────────────────────────────────────────

if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")
    from langchain_community.vectorstores import FAISS

    vector_db = FAISS.load_local(
        INDEX_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
else:
    documents = load_pdfs_from_directory(PDF_FOLDER)
    if not documents or len(documents) == 0:
        st.error("No PDFs found or processed. Please check the travel_data folder.")
        st.stop()

    from langchain_core.documents import Document

    docs = [Document(page_content=content, metadata={"source": fn})
            for fn, content in documents.items()]

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    from langchain_community.vectorstores import FAISS

    print("Creating FAISS index...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}")

retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# ────────────────────────────────────────────────
# LLM
# ────────────────────────────────────────────────

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    num_ctx=8192,
)

# ────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────

from langchain_core.prompts import PromptTemplate

prompt_template = """\
You are IndiaTravelGuru, an expert travel assistant specialized in travel planning within India.

Retrieved context (use it when relevant):
{context}

Current date: {current_date}
User location assumption: {user_location}

Conversation history:
{history}

User question: {question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "history", "current_date", "user_location"]
)

# ────────────────────────────────────────────────
# Chain helpers
# ────────────────────────────────────────────────

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def get_context(question: str) -> str:
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


def get_history(_) -> str:
    messages = st.session_state.get("messages", [])
    if not messages:
        return "(No previous conversation)"

    # Exclude the very last message (current user input)
    formatted = []
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n\n".join(formatted) or "(No previous conversation)"


# ────────────────────────────────────────────────
# The RAG chain – fixed version
# ────────────────────────────────────────────────

rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(get_context),
            "history": RunnableLambda(get_history),
            "current_date": lambda _: date.today().strftime("%Y-%m-%d"),
            "user_location": lambda _: "India",  # you can also read from st.session_state if you want
        }
        | prompt
        | llm
        | StrOutputParser()
)

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────

st.title("India Travel Guru Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = """
Hello! I'm IndiaTravelGuru, your expert travel planner for India.

To get the best personalized itinerary, please enter your query in this format:  
**Plan a [duration]-day trip to [destination] for [group size and type, e.g., family of 4] in [month year] with [budget level, e.g., moderate].**

For example: "Plan a 10-day trip to Rajasthan for a family of 4 in December 2025 with moderate budget."

How can I help you plan your trip?
    """
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Enter your travel query:"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})