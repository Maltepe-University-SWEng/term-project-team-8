# 🇹🇷 Turkish Joke Generator – AI Powered Joke App

An interactive web application that generates short and funny Turkish jokes using a fine-tuned language model. Built with a FastAPI backend and a lightweight responsive frontend, the system offers instant joke creation based on fixed prompting and post-processing techniques.

---

## 📌 Project Overview

This project focuses on generating culturally relevant, short Turkish jokes using natural language generation. A LoRA fine-tuned GPT-2 model (`ytu-ce-cosmos/turkish-gpt2-large`) was used and further refined with custom data. The system applies blacklist filtering and joke polishing using a base model to ensure cleaner outputs.

Users interact through a simple web interface with a single button. Each click generates a brand-new joke within seconds.

---

## 🚀 Features

- 🤖 LoRA fine-tuned transformer model
- 🔁 Automatic retry logic for invalid outputs
- 🧼 Blacklist filtering for safety
- 🧠 Base model refinement for better punchlines
- 💻 Responsive frontend with visual characters
- 🔧 API built with FastAPI
- 📊 Manual test cases and validation steps

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/turkish-joke-generator
cd turkish-joke-generator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Hugging Face login

```bash
huggingface-cli login
```

### 4. Run the backend

```bash
uvicorn app:app --reload
```

### 5. Open the frontend

Use Live Server or open `frontend/index.html` in your browser.

---

## 🧪 Example Output

```
Temel sınavda hiçbir şey yazmamış. Öğretmen: “Bu ne?”  
Temel: “Sessiz protesto.”
```

---

## 🧠 Technologies Used

- Python 3.10+
- FastAPI
- Hugging Face Transformers
- PEFT (LoRA)
- PyTorch (MPS optimized)
- HTML / CSS / JavaScript
- Google Colab & Kaggle for training
- Trello & GitHub for project management

---

## 👥 Team

| Name       | Role                                  |
|------------|----------------------------------------|
| Melih      | Model training, backend & frontend dev |
| Hüseyin    | Tech research, data prep, UI support   |
| Nurşeyda   | UI design, Figma, frontend dev         |
| Onur       | Model support, data cleaning           |
| İdil       | Social context & frontend help         |
| Afnan      | Visual design, data gathering          |

---

## 📁 Project Structure

```
.
├── app.py                  # FastAPI backend
├── requirements.txt
├── frontend/
│   ├── index.html
│   └── img/                # Character images
├── GPT2/                   # Fine-tuned LoRA adapter
└── README.md
```

---

## 📄 License

This project is for academic purposes only. All datasets used are curated and cleaned manually for research use.
