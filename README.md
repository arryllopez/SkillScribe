# 🚀 BuzzMatch (Resume Keyword Extractor)

BuzzMatch is an AI-powered tool that fine-tunes open-source LLMs to extract and generate **resume buzzwords** from job descriptions.  
It helps job seekers align their resumes with recruiter expectations by surfacing **key skills, technologies, and keywords**.  

---

## 📌 Features
- Extracts relevant keywords from job postings.  
- Generates **resume-ready buzzwords** (e.g., `AWS`, `Kubernetes`, `Data Analysis`).  
- Fine-tuned using **LoRA adapters** for efficient training.  
- Supports **4-bit/8-bit quantization** with `bitsandbytes` for low-memory training.  
- Built with Hugging Face `transformers`, `datasets`, and `peft`.  

---

## ⚙️ Installation

Make sure you have **Python 3.9+** and **CUDA 11.8/12.1** installed.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate bitsandbytes peft evaluate scipy
