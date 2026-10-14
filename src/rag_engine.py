import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Muat environment variable dari file .env
load_dotenv()

# Konfigurasi direktori
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

def get_embedding_model():
    """Menggunakan embedding model lokal yang ringan dan cepat (all-MiniLM-L6-v2)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_active_groq_model(api_key: str) -> str:
    """Mendeteksi model LLM chat terbaik yang sedang aktif di server Groq secara otomatis."""
    client = Groq(api_key=api_key)
    models = client.models.list()
    available_ids = [
        m.id for m in models.data 
        if "whisper" not in m.id.lower() and "guard" not in m.id.lower() and "tts" not in m.id.lower()
    ]
    
    # Daftar urutan prioritas model terbaik jika tersedia
    preferred_keywords = ["llama-4", "llama-3.3", "llama-3.1", "llama", "qwen", "mixtral", "gemma"]
    for kw in preferred_keywords:
        for m_id in available_ids:
            if kw in m_id.lower():
                return m_id
                
    return available_ids[0]

def build_vector_store():
    """Membaca semua PDF di folder data/, melakukan chunking, dan menyimpannya ke ChromaDB."""
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise ValueError(f"Folder '{DATA_DIR}' kosong. Masukkan minimal satu file PDF terlebih dahulu.")

    print("[INFO] Membaca dokumen PDF dari folder data/...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError("Tidak ada teks yang dapat diekstrak dari dokumen PDF.")

    print(f"[INFO] {len(raw_docs)} halaman dokumen berhasil dimuat.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"[INFO] Dokumen berhasil dipecah menjadi {len(chunks)} text chunks.")

    print("[INFO] Membuat vector embeddings & menyimpan ke ChromaDB...")
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    print("[SUCCESS] Vector store berhasil dibuat dan disimpan di folder 'vectorstore/'.")
    return vectorstore

def get_rag_chain():
    """Menyiapkan retrieval chain menggunakan ChromaDB dan LLM Groq."""
    embeddings = get_embedding_model()
    
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY belum diisi di dalam file .env!")

    # Pilih model aktif secara otomatis agar tidak terkena error 404 model_not_found
    active_model = get_active_groq_model(groq_api_key)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=active_model,
        temperature=0.2
    )

    system_prompt = (
        "Anda adalah asisten teknis cerdas berbasis AI untuk dokumen enterprise. "
        "Gunakan potongan konteks yang diberikan di bawah ini untuk menjawab pertanyaan pengguna secara akurat, "
        "profesional, dan jelas. Jika jawabannya tidak ditemukan di dalam konteks, katakan bahwa informasi tersebut "
        "tidak tersedia di dalam dokumen internal, jangan mengarang jawaban sendiri.\n\n"
        "Konteks Dokumen:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    build_vector_store()