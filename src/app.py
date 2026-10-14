import streamlit as st
import os
from rag_engine import get_rag_chain, build_vector_store

st.set_page_config(
    page_title="Enterprise Technical Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise Technical Document Assistant")
st.caption("RAG-Powered AI Assistant untuk Tanya-Jawab Dokumen Teknis Secara Real-Time")

# Sidebar untuk manajemen indexing
with st.sidebar:
    st.header("⚙️ Konfigurasi Basis Pengetahuan")
    st.write("Klik tombol di bawah jika kamu baru menambahkan atau memperbarui file PDF di folder `data/`.")
    
    if st.button("🔄 Bangun / Refresh Vector Store"):
        with st.spinner("Membaca dokumen PDF & membuat embedding..."):
            try:
                build_vector_store()
                st.success("Vector store berhasil diperbarui!")
            except Exception as e:
                st.error(f"Gagal memperbarui vector store: {e}")

# Inisialisasi riwayat chat di Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya adalah asisten dokumen teknis. Ada yang bisa saya bantu terkait isi dokumen di sistem?"}
    ]

# Tampilkan seluruh riwayat percakapan
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Tangani input pertanyaan dari pengguna
if user_query := st.chat_input("Ketik pertanyaan seputar dokumen teknis di sini..."):
    # Tampilkan pesan pengguna di layar
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Proses pencarian dan pembuatan jawaban via RAG Engine
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Menganalisis dokumen & menyusun jawaban..."):
            try:
                rag_chain = get_rag_chain()
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                # Tampilkan sumber referensi potongan dokumen (metadata context)
                sources = response.get("context", [])
                if sources:
                    source_names = list(set([os.path.basename(doc.metadata.get("source", "Dokumen")) for doc in sources]))
                    answer += f"\n\n---\n📌 **Sumber Dokumen Terkait:** {', '.join(source_names)}"

                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as err:
                error_msg = f"Terjadi kendala saat memproses jawaban: {err}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})