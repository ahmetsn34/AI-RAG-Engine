# AI-RAG-Engine 🚀

A premium, enterprise-ready **Local RAG (Retrieval-Augmented Generation)** Orchestrator built with FastAPI, FAISS, and Sentence Transformers. It allows companies to interact with their confidential files (PDF, Excel, TXT) fully offline using localized LLMs via a dynamically configurable interface.

---

## ✨ Key Features

* **Multi-Format Document Ingestion:** Native parsing support for **PDF**, **Excel (`.xlsx`, `.xls`)**, and **TXT** files.
* **Vector-Based Information Retrieval:** Uses `SentenceTransformer` (`all-MiniLM-L6-v2`) and Facebook's **FAISS** for highly accurate semantic proximity search.
* **ChatGPT-Style Streaming:** Leverages FastAPI's `StreamingResponse` to stream token outputs dynamically to the frontend for a premium user experience.
* **Contextual Chat Memory:** Retains a rolling window of recent conversation history, ensuring the local LLM maintains conversational context.
* **Dynamic Network Adaptability:** Features an integrated **Settings Modal** storing configurations in `localStorage`. Users can dynamically update the local LLM server IP address and model schema on the fly.
* **Eye-Comfort Premium UI:** An ultra-modern, minimalist dark-theme user interface inspired by Apple and Vercel design standards, crafted with custom typography (`Plus Jakarta Sans`).

---

## 🛠️ Architecture & Tech Stack

* **Backend Framework:** FastAPI (Python)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings Model:** HuggingFace `all-MiniLM-L6-v2`
* **Data Processing:** Pandas & OpenPyXL (for Excel processing), PyPDF2 (for PDF text-extraction)
* **Local LLM Host:** Ollama (Compatible with Qwen2.5, Llama3, etc.)

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python installed, along with your local LLM serving framework (e.g., Ollama running on a local server or container network).

### 2. Installation
Clone the repository and install the required corporate dependencies:
```bash
git clone [https://github.com/ahmetsn34/AI-RAG-Engine.git](https://github.com/ahmetsn34/AI-RAG-Engine.git)
cd AI-RAG-Engine
pip install fastapi uvicorn requests PyPDF2 pandas openpyxl langchain-text-splitters sentence-transformers faiss-cpu numpy
