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
        "Ты — полезный помощник МегаФона. Отвечай на русском кратко и точно. "
        "Если ответа нет в контексте, скажи, что не знаешь.\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    ),
    input_variables=["context", "question"],
)


def load_llm(model_name: str) -> HuggingFacePipeline:
    """
    Load a causal language model for generation.
    Uses accelerate to place the model on the available device.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    gen = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=gen)


def build_retrieval_qa_chain(vector_store: Chroma, llm: HuggingFacePipeline) -> RetrievalQA:
    """
    Create a RetrievalQA chain that retrieves only 'tariff' documents.
    """
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3},
        filter={"type": "tariff"}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT},
    )


def interactive_chat(chain: RetrievalQA) -> None:
    """
    Run the interactive REPL. User asks questions in Russian;
    the bot answers based on the RAG chain.
    """
    print("Добро пожаловать в чат-бот EVA! Задавайте вопросы о тарифах МегаФон.")
    while True:
        q = input("\nВведите ваш вопрос (или 'выход' для завершения): ").strip()
        if q.lower() in {"выход", "quit", "exit"}:
            print("До свидания!")
            break
        if not q:
            continue
        res = chain.invoke({"query": q})
        print("Ответ:", res.get("result", "К сожалению, я не смогла найти ответ."))


def main():
    parser = argparse.ArgumentParser(
        description="Run EVA chatbot using an existing Chroma index"
    )
    parser.add_argument(
        "--persist_dir",
        required=True,
        help="Directory of the persisted Chroma database",
    )
    parser.add_argument(
        "--model",
        default="t-tech/T-pro-it-2.0-AWQ",
        help="HuggingFace model for generation",
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="HuggingFace embedding model",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.persist_dir):
        print(f"Chroma directory '{args.persist_dir}' not found.")
        sys.exit(1)

    # Load existing Chroma vector store
    vectordb = Chroma(
        persist_directory=args.persist_dir,
        embedding_function=HuggingFaceEmbeddings(model_name=args.embedding_model),
        collection_name="megafon_tariffs",
    )

    # Prepare the RAG chain
    llm = load_llm(args.model)
    qa_chain = build_retrieval_qa_chain(vectordb, llm)

    # Start chatting
    interactive_chat(qa_chain)


if __name__ == "__main__":
    main()
