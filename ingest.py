import os
import pinecone
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv
import logging
import sys
from typing import List
from pinecone import ServerlessSpec

# --- Basic Configuration ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

# --- Load API Keys ---
try:
    load_dotenv()
    log.info("Loaded environment variables.")
except Exception as e:
    log.error(f"Could not load .env file: {e}")
    sys.exit(1)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    log.error("OpenAI or Pinecone API key not found in .env file. Please check.")
    sys.exit(1)


# --- This class now adds BOTH 'author' and 'work' metadata ---
class AddCustomMetadata(TransformComponent):
    def __call__(self, nodes: List[Document], **kwargs) -> List[Document]:
        for node in nodes:
            file_path = node.metadata.get("file_path", "")
            file_name = node.metadata.get("file_name", "")

            # 1. Add Author Tag from folder name
            try:
                path_parts = os.path.normpath(file_path).split(os.sep)
                author_index = path_parts.index("data") + 1
                if author_index < len(path_parts):
                    node.metadata["author"] = path_parts[author_index]
                else:
                    node.metadata["author"] = "Unknown"
            except (ValueError, IndexError) as e:
                log.warning(f"Could not extract author for file {file_path}: {e}")
                node.metadata["author"] = "Unknown"

            # 2. Add Work Tag from file name
            # This cleans up the filename to be a readable title.
            # E.g., "the_prince.pdf" -> "The Prince"
            work_title = os.path.splitext(file_name)[0] # remove .pdf
            work_title = work_title.replace("_", " ").replace("-", " ").title()
            node.metadata["work"] = work_title

        return nodes


def main():
    log.info("Loading documents from the ./data folder...")
    reader = SimpleDirectoryReader("./data", recursive=True)
    documents = reader.load_data()
    log.info(f"Loaded {len(documents)} document(s).")

    INDEX_NAME = "holodeck-index"
    EMBED_DIMENSION = 1536

    log.info("Connecting to Pinecone...")
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

    log.info(f"Checking if index '{INDEX_NAME}' exists...")
    if INDEX_NAME not in pc.list_indexes().names():
        log.info(f"Index '{INDEX_NAME}' does not exist. Creating it now...")
        pc.create_index(
            name=INDEX_NAME, dimension=EMBED_DIMENSION, metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        log.info(f"Index '{INDEX_NAME}' created successfully.")
    else:
        log.info(f"Index '{INDEX_NAME}' already exists.")

    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    log.info("Setting up ingestion pipeline...")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=100),
            AddCustomMetadata(), # Use our new, more powerful class
            OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY),
        ],
        vector_store=vector_store,
    )

    log.info("Starting ingestion... This will rebuild your index with 'work' metadata.")
    pipeline.run(documents=documents, show_progress=True, num_workers=4)

    log.info("\n---")
    log.info("All documents have been re-ingested into Pinecone with author and work metadata.")
    log.info("---")

if __name__ == '__main__':
    main()