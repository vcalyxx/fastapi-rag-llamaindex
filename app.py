from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core import SimpleDirectoryReader

## Connection to the Qdrant client
client = QdrantClient(url="http://localhost:6333")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="fastapi_docs"
)

## Load documents
documents = SimpleDirectoryReader(
    "docs/docs/en/docs",
    required_exts=[".md"]
).load_data()

print(f"Documents loaded: {len(documents)}")

