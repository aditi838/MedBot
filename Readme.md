# MedBot – AI Healthcare Chatbot 🤖🩺

MedBot is an AI-powered healthcare chatbot designed to provide users with **reliable health information** through natural, empathetic conversations. It combines **context awareness**, **retrieval-augmented generation (RAG)**, and **real-time web search** to assist users with common health-related questions.

---

## 🚀 Features

- 💬 **Context-aware Conversations**: Remembers recent dialogue to give meaningful responses.
- 🧠 **Intent Detection & Topic Extraction**: Understands user queries and extracts key medical topics.
- 📚 **WHO-based Document Retrieval**: Uses RAG pipeline over WHO health documents.
- 🌐 **Real-time Web Search**: Integrates Tavily API to fetch current and relevant health updates.
- 💡 **Empathetic Response Generation**: Prioritizes user mental wellness through tone-aware replies.

---

## 🛠️ Tech Stack

| Tool              | Purpose                             |
|-------------------|-------------------------------------|
| **Python**        | Core backend logic                  |
| **FastAPI**       | Backend API development             |
| **Streamlit**     | Frontend interface                  |
| **LangChain**     | RAG pipeline, agents, and memory    |
| **ChromaDB**      | Vector storage for document retrieval |
| **Groq-hosted LLMs** | Fast and efficient language generation |
| **Tavily API**    | Real-time search and web summarization |

---

## 📷 Screenshots

| Chat Interface |
|----------------|
| ![MedBot UI](![image](https://github.com/user-attachments/assets/c5bab14f-4c41-4260-9436-15827e4aae12)
) |

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medbot.git
   cd medbot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   - Start **FastAPI backend**:
     ```bash
     uvicorn backend.main:app --reload
     ```
   - Launch **Streamlit frontend**:
     ```bash
     streamlit run Home.py
     ```

---

## 📁 Project Structure

```
MedBot/
├── Home.py                      # Streamlit app interface
├── ai_agent.py                  # LangChain agent logic for query handling
├── backend.py                   # Backend control and coordination(using FastAPI)
├── evaluator.py                 # Evaluation logic and RAG metrics
├── eval_WHO_QnA_Pairs.py        # Evaluation script for WHO Q&A pairs
├── eval_QnA_Visuals.py          # Visualization of evaluation results
├── who_scraper.py               # Scraper for WHO Q&A content
├── requirements.txt             # Python dependencies

# Data & Evaluation Files
├── debug_results.csv
├── disease_qna_pairs.csv
├── evaluation_results_api_new.csv
├── low_performance_cases.csv
├── matched_factsheets_qna.csv
├── sample_eval_dataset.csv

└── README.md                    # Project documentation


---

## 🧠 Future Improvements

- Add voice input support  
- Enable user authentication for saving chat history  
- Integrate more health datasets for broader coverage  
- Add feedback collection mechanism

---

## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Tavily Search](https://www.tavily.com/)
- [Groq Cloud](https://console.groq.com/)

---

> 💡 *MedBot is an academic/educational prototype. It is not a replacement for professional medical advice.*
