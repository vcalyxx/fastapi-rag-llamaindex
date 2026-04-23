from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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

## Embedding

# Initialize embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en"
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

