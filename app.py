from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

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


## Chunking

parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

nodes = parser.get_nodes_from_documents(documents)




print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(nodes)}")
print("\nSample chunk:\n")
print(nodes[0].text[:500])
