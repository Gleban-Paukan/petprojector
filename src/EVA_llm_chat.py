"""
Interactive RAG chatbot for EVA using Chroma and a local LLM.

Usage:
    python eva_chatbot.py --persist_dir ./chroma_db \
                         --model HuggingFaceTB/SmolLM3-3B \
                         --embedding_model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
"""

import argparse
import os
import sys

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Prompt template for RetrievalQA
PROMPT = PromptTemplate(
    template=(
        "You are MegaFon’s helpful assistant. Answer concisely and accurately. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    input_variables=["context", "question"],
)


def load_llm(model_name: str) -> HuggingFacePipeline:
    """
    Load a causal language model for generation.
    Uses accelerate to place model on available device.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


def main():
    """Load the Chroma index, set up RetrievalQA, and run interactive loop."""
    parser = argparse.ArgumentParser(description="Run EVA RAG chatbot")
    parser.add_argument(
        "--persist_dir",
        default="./chroma_db",
        help="Directory where Chroma index is stored",
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM3-3B",
        help="HuggingFace model name for generation",
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="HuggingFace embedding model name",
    )
    args = parser.parse_args()

    vectordb = Chroma(
        persist_directory=args.persist_dir,
        embedding_function=HuggingFaceEmbeddings(model_name=args.embedding_model),
        collection_name="megafon_tariffs",
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=load_llm(args.model),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
    )

    print("EVA-GPT is ready. Type 'выход' to exit.")
    while True:
        q = input("Question: ").strip()
        if q.lower() in {"выход", "quit", "exit"}:
            break
        result = chain.invoke({"query": q})
        print("Answer:", result.get("result", "[no answer]"))


if __name__ == "__main__":
    main()