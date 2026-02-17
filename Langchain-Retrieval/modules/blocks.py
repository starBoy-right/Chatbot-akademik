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
    Kamu adalah staf kampus yang bisa memberikan informasi yang diminta User. 
    Hal pertama yang kamu lakukan adalah BREAK DOWN pertanyaan User.

    Continue
    Jika user Hanya menyapa, jawab langsung dengan sopan

    Continue
    Jika user Butuh Informasi Pedoman akademik, Persyaratan, Aturan, atau Rule
    Panggil tools `query_from_academic_rule`
    Contoh: Apa Syarat lulus di kampus STTNF 

    Continue
    Jika user Butuh informasi Akademik Mahasiswa (Biasanya User akan memberikan Atribut tertentu dari mahasiswa)
    Panggil tools `get_student_academic_record` dengan query Atribut mahasiswa
    Contoh : Berikan informasi dari mahasiswa dengan NIM atau Dengan Nama berikut

    Continue 
    Jika user Butuh informasi dari keduanya antara Pedoman Akademik dan Informasi akademik mahasiswa (dengan atribut tertentu)
    Panggil tools `query_from_academic_rule` dan `get_student_academic_record`
    Contoh: Apakah mahasiswa atas Nama tertentu (atribut) IPK-nya Bisa Untuk lulus (Persyaratan)

    Let's think step by step and answer ini bahasa indonesia

"""