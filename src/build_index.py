"""
Builds a Chroma vector store for Megafon tariffs and extra web pages.

Usage:
    python build_index.py --data megafon_tariffs.jsonl \
                          --urls_path urls.json \
                          --persist_dir ./chroma_db
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def load_tariff_data(jsonl_path: str) -> List[Document]:
    """Parse JSONL with tariff info and return a list of Documents."""
    docs: List[Document] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            parts = [
                f"Tariff {rec.get('name','')}",
                f"Price: {rec.get('price_rub','')} RUB",
                f"Data: {rec.get('data_gb','')} GB",
                f"Minutes: {rec.get('minutes','')}",
                f"SMS: {rec.get('sms','')}",
                f"Description: {rec.get('description','')}",
            ]
            text = ". ".join(p for p in parts if p)
            docs.append(
                Document(
                    page_content=text,
                    metadata={"type": "tariff", "name": rec.get("name", "")},
                )
            )
    return docs


def load_extra_urls(urls: List[str]) -> List[Document]:
    """Download each URL, split into chunks and return Documents of type 'extra'."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs: List[Document] = []

    for u in urls:
        try:
            print("Fetching:", u)
            r = requests.get(u, timeout=10)
            r.raise_for_status()
            raw = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
            for idx, chunk in enumerate(splitter.split_text(raw)):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"type": "extra", "url": u, "part": idx},
                    )
                )
        except Exception as e:
            print("Skip", u, "->", e)

    return docs


def build_index(
    all_docs: List[Document], embed_model: str, persist_dir: str
) -> None:
    """Split docs into chunks, build a Chroma index and persist it to disk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunked = splitter.split_documents(all_docs)

    vectordb = Chroma.from_documents(
        chunked,
        HuggingFaceEmbeddings(model_name=embed_model),
        persist_directory=persist_dir,
        collection_name="megafon_tariffs",
    )
    vectordb.persist()
    print("✔ Index saved to", persist_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma index for EVA RAG")
    parser.add_argument("--data", required=True, help="Path to megafon_tariffs.jsonl")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma directory")
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="HuggingFace embedding model",
    )
    parser.add_argument(
        "--urls_path",
        default="urls.json",
        help="JSON file with extra URLs to crawl",
    )
    args = parser.parse_args()

    with open(args.urls_path, "r", encoding="utf-8") as f:
        extra_urls = json.load(f)

    documents = load_tariff_data(args.data)
    documents.extend(load_extra_urls(extra_urls))
    build_index(documents, args.embedding_model, args.persist_dir)
