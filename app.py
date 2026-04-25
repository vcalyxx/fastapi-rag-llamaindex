import os

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fastapi_docs")

DOCS_PATH = os.getenv("DOCS_PATH", "docs/docs/en/docs")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "2"))

## Connection to the Qdrant client
client = QdrantClient(url=QDRANT_URL)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION
)

## Load documents
documents = SimpleDirectoryReader(
    DOCS_PATH,
    required_exts=[".md"]
).load_data()


## Chunking

parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

nodes = parser.get_nodes_from_documents(documents)

## Embedding

# Initialize embedding model
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# Create Index
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)


llm = HuggingFaceLLM(
    model_name=LLM_MODEL,
    tokenizer_name=LLM_MODEL,
    max_new_tokens=MAX_NEW_TOKENS
)

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=SIMILARITY_TOP_K
)

response = query_engine.query("How do I create a FastAPI app?")

print(response.response)