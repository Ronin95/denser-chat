import os
import glob
import shutil
import json
import logging
from urllib.parse import urlparse, urlencode, quote
from denser_retriever.retriever import DenserRetriever
from denser_retriever.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, index_name):
        self.index_name = index_name
        self.retriever = DenserRetriever(
            index_name=index_name,
            keyword_search=ElasticKeywordSearch(
                top_k=100,
                es_connection=create_elasticsearch_client(
                    url="http://localhost:9200",
                    username="elastic",
                    password="",
                ),
                drop_old=True,
                analysis="default"
            ),
            vector_db=None,
            reranker=None,
            embeddings=None,
            gradient_boost=None,
            search_fields=["annotations:keyword"],
        )
        # Batch size for ingestion
        self.ingest_bs = 2000

    def index(self, docs_file):
        logger.info(f"== Ingesting file {docs_file}")
        if not os.path.exists(docs_file):
            logger.error(f"File {docs_file} does not exist.")
            return

        with open(docs_file, "r", encoding="utf-8") as f:
            docs = []
            num_docs = 0
            for line in f:
                try:
                    doc_dict = json.loads(line)
                    # Transform the "annotations" field to a JSON string if present.
                    if "annotations" in doc_dict:
                        doc_dict["annotations"] = json.dumps(doc_dict["annotations"])
                    docs.append(Document(**doc_dict))
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
                    continue

                if len(docs) >= self.ingest_bs:
                    self.retriever.ingest(docs, overwrite_pid=True)
                    num_docs += len(docs)
                    logger.info(f"Ingested {num_docs} documents")
                    docs = []
            if docs:
                self.retriever.ingest(docs, overwrite_pid=True)
                num_docs += len(docs)
                logger.info(f"Ingested {num_docs} documents")

        size = os.path.getsize(docs_file)
        logger.info(f"File {docs_file} size: {size} bytes")

    def retrieve(self, query, top_k, meta_data):
        passages = self.retriever.retrieve(query, top_k, meta_data)
        return passages


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Process PDFs and HTML pages listed in sources file and create an index."
    )
    parser.add_argument(
        'sources_file',
        type=str,
        help="Path to the sources.txt file containing list of PDFs and URLs"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Directory where output files will be stored"
    )
    parser.add_argument(
        'index_name',
        type=str,
        help="Name for the index to be created"
    )
    args = parser.parse_args()

    # Assume build.py produces a combined passages file named 'passages.jsonl' inside the output directory.
    combined_file = os.path.join(args.output_dir, "passages.jsonl")
    idx = Indexer(args.index_name)
    idx.index(combined_file)
