"""

ini adalah module untuk agent RAG yang dipakai, seperti model LLM, model embedding, chromadb, etc
"""

from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from transformers import BertTokenizer, AutoModel
from typing import List
from langchain_core.embeddings import Embeddings
import torch


# model LLM
llm_model = ChatOllama(
    model='mistral-openorca:7b-q4_0',
    temperature=0,
    streaming=True,
)


# model embedding
class IndoBertEmbeddings(Embeddings):
    def __init__(self, model_name="indobenchmark/indobert-base-p1"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()


    def _generate_embedding(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state

        sentence_embeddings = token_embeddings.mean(dim=1)

        return sentence_embeddings.squeeze().tolist()
    

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_embedding(text) for text in texts]
    

    # pencarian query
    def embed_query(self, text: str) -> List[float]:
        return self._generate_embedding(text)

embeddings_model = IndoBertEmbeddings()


# vector store -> chromaDB
vector_store = Chroma(
    collection_name="chroma-with-langchain",
    embedding_function=embeddings_model,
    persist_directory="../Langchain-Project/notebooks/chroma_langchain_db"
)