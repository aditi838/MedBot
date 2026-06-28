# MedBot — RAG Healthcare Assistant

MedBot is a Retrieval-Augmented Generation (RAG) healthcare assistant that answers health-related questions using WHO knowledge, semantic search, intent-aware routing, and conversation memory. It uses Groq-hosted Llama 3 70B for response generation and Tavily for real-time health lookups.

> **Disclaimer:** MedBot is an academic prototype. It is not a substitute for professional medical advice.

---

## How it works

Every user query follows a deterministic pipeline before reaching the language model:

```
User Query
    │
    ▼
Intent Classification  (6 classes)
    │
    ├── medical_fact      →  WHO RAG (ChromaDB retrieval)
    ├── specific_symptom  →  WHO RAG (ChromaDB retrieval)
    ├── follow_up_detail  →  Conversation memory + WHO RAG
    ├── vague_symptom     →  Clarification workflow
    ├── real_time         →  Tavily Search
    └── general_chat      →  Conversational LLM chain
    │
    ▼
Groq · Llama 3 70B
    │
    ▼
Grounded Response
```

Routing ensures each query type gets a specialised chain rather than a single generic pipeline — reducing unnecessary retrieval and improving response quality.

---

## Knowledge base

- **31 WHO health pages** scraped and cleaned with a custom BeautifulSoup pipeline
- **7,503 semantic chunks** indexed in ChromaDB using recursive character chunking (500 chars, 50 overlap)
- **SentenceTransformer embeddings** for semantic similarity retrieval

---

## Evaluation

A custom local evaluation framework was built to assess response quality without an external LLM judge:

| Metric | Method |
|---|---|
| Semantic Similarity | Cosine similarity between generated and WHO reference answers (SentenceTransformer) |
| Keyword Overlap | Factual term coverage between generated and reference responses |
| Grounding Score | Whether generated claims are supported by retrieved WHO context |

Evaluated on **231 benchmark questions** spanning all six intent categories.

---

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Groq · Llama 3 70B |
| Orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | SentenceTransformers |
| Real-time search | Tavily API |
| Scraping | BeautifulSoup |

---

## Project structure

```
MedBot/
├── Home.py                       # Streamlit frontend
├── backend.py                    # FastAPI backend
├── ai_agent.py                   # LangChain intent routing and chain logic
├── who_scraper.py                # WHO page scraper and cleaner
├── evaluator.py                  # Evaluation metrics (similarity, grounding, overlap)
├── eval_WHO_QnA_Pairs.py         # Evaluation runner
├── eval_QnA_Visuals.py           # Evaluation result visualisations
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
│
└── data/                         # Evaluation outputs
    ├── disease_qna_pairs.csv
    ├── evaluation_results_api_new.csv
    ├── low_performance_cases.csv
    ├── matched_factsheets_qna.csv
    └── sample_eval_dataset.csv
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/aditi838/MedBot.git
cd MedBot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

- Get a Groq key at [console.groq.com](https://console.groq.com)
- Get a Tavily key at [tavily.com](https://www.tavily.com)

### 5. Run the app

Start the FastAPI backend:

```bash
uvicorn backend:app --reload
```

In a separate terminal, launch the Streamlit frontend:

```bash
streamlit run Home.py
```

---

## Running the evaluation

```bash
python eval_WHO_QnA_Pairs.py     # Run evaluation against WHO Q&A benchmark
python eval_QnA_Visuals.py       # Generate visualisation of results
```

---

## Key engineering decisions

**Why RAG?** LLMs hallucinate on medical questions. Grounding responses in retrieved WHO documents reduces that risk.

**Why recursive chunking?** Fixed-size chunking split medical sentences mid-thought, degrading retrieval. Recursive chunking respects sentence and paragraph boundaries.

**Why intent routing?** A single chain cannot cleanly handle medical facts, vague symptoms, follow-ups, and general chat simultaneously. Routing each to a specialised chain improved quality and reduced unnecessary retrieval.

**Why shared memory?** Isolated per-chain memory objects silently break multi-turn conversations. A single shared `ConversationBufferMemory` instance passed into all chains keeps context consistent.

---

## Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Groq Cloud](https://console.groq.com/)
- [Tavily Search](https://www.tavily.com/)
- [WHO Health Topics](https://www.who.int/health-topics)
