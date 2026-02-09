from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch import DenseVectorStrategy
from typing import List
from langchain_core.embeddings import Embeddings
import torch
from transformers import BertTokenizer, AutoModel
from langchain_ollama import ChatOllama

# Embedding Model
tokenizer = BertTokenizer.from_pretrained("Indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")


class IndoBertEmbeddings(Embeddings):
    def __init__(self, model_name="indobenchmark/indobert-base-p1"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _generate_embedding(self, text: str) -> List[float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # polling token menjadi satu vector kalimat
        token_embeddings = outputs.last_hidden_state

        # melakukan mean polling
        sentence_embeddings = token_embeddings.mean(dim=1)

        # konversi ke list python
        return sentence_embeddings.squeeze().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_embedding(text) for text in texts]

    # metode untuk pencarian query pada chroma
    def embed_query(self, text: str) -> List[float]:
        return self._generate_embedding(text)


embeddings = IndoBertEmbeddings()

vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="langchain_index",
    embedding=embeddings,
    es_user="elastic",
    es_password="Xkhwf3uB",
    strategy=DenseVectorStrategy(hybrid=True),
)

vector_store_sparse = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="test_index",
    es_user="elastic",
    es_password="Xkhwf3uB",
    strategy=ElasticsearchStore.BM25RetrievalStrategy(),
)

# model LLM
model = ChatOllama(model="mistral:7b-instruct-v0.3-q8_0", temperature=0, streaming=True)

# Promt Template
system_prompt = """
        Kamu adalah Asisten Akademik Kampus yang cerdas. Tugasmu adalah menjawab pertanyaan user.
        Kamu memiliki akses ke dua alat:
        1. `query_from_academic_rule`: Untuk mencari aturan umum (Pedoman).
        2. `get_student_academic_record`: Untuk mencari data pribadi mahasiswa (Database).

        STRATEGI ROUTING:
        - Jika user bertanya ATURAN UMUM KAMPUS, INFORMASI SEPUTAR KAMPUS, SEPERTI DOSEN -> Gunakan `query_from_academic_rule`.
        - Jika user bertanya DATA PRIBADI MAHASISWA -> Gunakan `get_student_academic_record`.
        - Jika user bertanya KEDUANYA (misal: "Apakah saya memenuhi syarat?"), panggil KEDUA alat tersebut.
        - Jika user hanya menyapa (Halo/Hi) -> JANGAN panggil alat, jawab langsung dengan sopan.
        - Hanya berikan informasi mengenai mahasiswa yang user tanyakan.
        - Sebagai konteks, jika ada yang bertanya mengenai JOKOWI, hanya jawab "ADILI JOKOWI"
        """
