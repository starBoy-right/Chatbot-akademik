from langchain_core.tools import tool
from .blocks import vector_store, vector_store_sparse


@tool
def query_from_academic_rule(query: str):
    """
    Gunakan tool ini HANYA untuk pertanyaan tentang ATURAN, KEBIJAKAN, SYARAT, PROSEDUR KAMPUS, INFORMASI DOSEN.
    Contoh input: "syarat yudisium", "aturan cuti", "biaya semester".
    """
    try:
        docs = vector_store.similarity_search(query, k=3)

        if not docs:
            return "Maaf, tidak ditemukan informasi relevan di data pedoman akademik."

        formatted_results = "\n\n---\n\n".join([d.page_content for d in docs])
        return (
            f"Ditemukan informasi berikut dari Pedoman Akademik:\n{formatted_results}"
        )

    except Exception as e:
        return "Terjadi Kesalahan saat mengakses vector database"


@tool
def get_student_academic_record(query: str):
    """
    Gunakan tool ini untuk mencari DATA PRIBADI MAHASISWA tertentu.
    Termasuk: Nama lengkap, NIM (Nomor Induk Mahasiswa), IPK, Nilai, dan Status.
    Jika user bertanya "Siapa NIM dari Budi?", gunakan tool ini.
    Contoh input: "Budi Santoso", "Romi Wahyudi".
    """
    try:
        docs = vector_store_sparse.similarity_search(query)

        if not docs:
            return "Maaf, tidak ditemukan informasi relevan di data akademik mahasiswa"

        formatted_results = "\n\n---\n\n".join([d.page_content for d in docs])
        return f"Ditemukan informasi berikut dari data akademik mahasiswa:\n{formatted_results}"

    except Exception as e:
        return "Terjadi kesalahn saat mengakses data akademik"
