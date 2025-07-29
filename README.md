**README.md**

```markdown
# EVA Chatbot RAG System

A local Retrieval-Augmented Generation (RAG) chatbot for Megafon tariffs.  
It builds a Chroma vector index from official tariff data and additional web pages, then answers user queries by retrieving the most relevant chunks and feeding them into a local LLM.

## Features

- **Index builder** (`build_index.py`):  
  - Loads `megafon_docs.jsonl` (tariff data) and a list of URLs (`urls.json`).  
  - Splits every document into 800-char chunks with 100-char overlap.  
  - Embeds chunks with a HuggingFace model.  
  - Persists Chroma database to disk.

- **Interactive chatbot** (`EVA_llm_chat.py`):  
  - Loads the persisted Chroma index.  
  - Uses a local HuggingFace LLM (4-bit/8-bit quantized if available).  
  - Retrieves top-k relevant chunks and feeds them into the model with a concise prompt template.  
  - Runs in a simple REPL (`Question:` → `Answer:`).

## Directory Layout

```



````

## Prerequisites

- **Python 3.12+**  
- **NVIDIA GPU (optional)** for faster embedding & generation.  
  - Without GPU, all operations fall back to CPU.

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/pet_projector.git
   cd pet_projector
````

2. **Create & activate a virtual environment**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Building the Index

```bash
python src/build_index.py \
  --data data/megafon_docs.jsonl \
  --urls_path data/urls.json \
  --persist_dir chroma_db
```

* After running, your Chroma database will live under `chroma_db/`.

## Running the Chatbot

```bash
python src/EVA_llm_chat.py \
  --persist_dir ./chroma_db \
  --model t-tech/T-pro-it-2.0-AWQ \
  --embedding_model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

* At the `Question:` prompt type your query in Russian.
* Type `выход` (or `quit`/`exit`) to end.

---

## NVIDIA 5000 Series GPU Setup (Ubuntu/WSL)

Run these commands **before** installing Python packages to get CUDA 12.8:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8 python3.12-venv git

# Optional: clone and set up ComfyUI if you use it for other tasks
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python3 -m venv venv
source venv/bin/activate

# Install nightly PyTorch for CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Then install the rest of requirements
pip install -r requirements.txt
```

---

## Exapmles of responses

```
Q: сколько стоит iPhone 16 Pro
A: iPhone 16 Pro 256GB стоит 118999 рублей. iPhone 16 Pro 512GB стоит 159999 рублей. iPhone 16 Pro 1TB стоит 199999 рублей. iPhone 16 Pro 1TB Max стоит 239999 рублей.
```

```
Q: сколько стоит тариф Минимальный
A: Тариф минимальный стоит 750 рублей.
```

```
Q: какой тариф самый дешевый в плане цены рублей за гигабайт 
A: Тариф Минимум. Цена за гигабайт 750 / 35 = 21.4 руб/Гб.
```

```
Q: какая IT компания самая лучшая в России?
A: Яндекс
```
