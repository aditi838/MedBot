# ai_agent.py (updated retrieval heuristic)
from dotenv import load_dotenv
import os
import re
from sentence_transformers import SentenceTransformer, util
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from tavily import TavilyClient
import torch
from difflib import SequenceMatcher
import pysqlite3 as pysqlite3_mod
import sys
sys.modules["sqlite3"] = pysqlite3_mod

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Embedding and vector store setup
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    embedding_function=embedder,
    persist_directory="chroma_store"
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
_ = retriever.get_relevant_documents("warmup")  # pre-warm index
similarity_model = SentenceTransformer(EMBED_MODEL)

# Updated system prompt with empathy
system_prompt = (
    "You are MedBot, a friendly and empathetic medical assistant. "
    "When the user describes vague symptoms, apologize for their discomfort "
    "and guide them to share onset, location, and severity. "
    "Never diagnose or prescribe. Recommend seeing a real doctor if needed."
)

# Prompt templates
EXTRACT_TOPIC_PROMPT = PromptTemplate(
    input_variables=["message", "history"],
    template="""
Extract the single-word (or short) medical topic from the user’s message and history.
Only output the topic name itself (e.g. "dengue", "malaria"), in lowercase.
If there is no clear topic, output exactly "unknown".
Message: {message}
History:
{history}
Topic:
"""
)

CLASSIFY_INTENT_PROMPT = PromptTemplate(
    input_variables=["message", "history"],
    template="""
Classify the intent of this user message into one of: vague_symptom, specific_symptom, medical_fact, real_time, general_chat, follow_up_detail.
Message: {message}
History:
{history}
Intent:
"""
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful medical assistant. Use the following WHO fact sheet information to answer the question clearly.
Provide your answer in a clear, structured format with bullet points or short paragraphs.

Context:
{context}

Question:
{question}

Answer:
"""
)

# Memory store
memory_store: dict[str, any] = {}
_stuff_chain_cache: dict[str, StuffDocumentsChain] = {}

# Tavily search
def get_tavily_search_tool():
    client = TavilyClient(api_key=TAVILY_API_KEY)
    def tavily_search(query: str) -> list[str]:
        result = client.search(query=query)
        return [res.get("content", "") for res in result.get("results", [])]
    return tavily_search

# Formatting and grounding helpers (unchanged)
def format_medical_search_result(raw_text: str) -> str:
    sections = {"summary": [], "statistics": [], "regions": [], "international": [], "deaths": [], "notes": [], "links": []}
    for line in raw_text.split("\n"):
        line = line.strip()
        lower = line.lower()
        if not line:
            continue
        elif any(k in lower for k in ["case","increase","decrease","report","percent","trend"]):
            sections["statistics"].append(line)
        elif any(k in lower for k in ["region","country","area","province","affected","endemic"]):
            sections["regions"].append(line)
        elif any(k in lower for k in ["international","cross-border","transmission","linked","travel"]):
            sections["international"].append(line)
        elif "death" in lower or "fatalit" in lower:
            sections["deaths"].append(line)
        elif any(k in lower for k in ["warning","shortage","vaccine","note","monitoring","action"]):
            sections["notes"].append(line)
        elif re.search(r"(http|www)\S+", line):
            sections["links"].append(line)
        else:
            sections["summary"].append(line)

    def fmt(title, items):
        return f"**{title}:**\n" + ("\n".join(items) if items else "N/A") + "\n"
    return (
        "🧾 **Health Update Summary**\n\n"
        + fmt("📝 Summary", sections["summary"])
        + fmt("📊 Statistics", sections["statistics"])
        + fmt("📍 Regions Affected", sections["regions"])
        + fmt("🌍 International Spread", sections["international"])
        + fmt("⚰️ Deaths", sections["deaths"])
        + fmt("📌 Notes", sections["notes"])
        + fmt("🔗 Sources", sections["links"])
    )

# Summarization helper

def summarize_results_with_llm(llm, snippets: list[str], query: str) -> str:
    prompt = PromptTemplate.from_template("""
You are a helpful medical assistant. Summarize the following search results related to the user's question.

Question:
{query}

Web Results:
{snippets}

Summary:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    snippet_text = "\n\n".join(snippets)
    return chain.run({"query": query, "snippets": snippet_text}).strip()


def best_overlap_ratio(answer: str, docs: list[str]) -> float:
    return max(SequenceMatcher(None, answer.lower(), d.lower()).ratio() for d in docs) if docs else 0.0

# Main response function

def get_response_from_ai_agent(session_id: str,
                               llm_id: str,
                               history: list[HumanMessage],
                               allow_search: bool,
                               system_prompt: str = system_prompt,
                               provider: str = "Groq"):
    if provider != "Groq":
        return {"response": "Only Groq provider is supported.", "source_tag": "error"}

    # Initialize memory
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = memory_store[session_id]
    for m in history:
        memory.chat_memory.add_message(m)

    # Instantiate LLM and chains
    llm = ChatGroq(model=llm_id)
    classify_chain = LLMChain(llm=llm, prompt=CLASSIFY_INTENT_PROMPT)
    extract_chain = LLMChain(llm=llm, prompt=EXTRACT_TOPIC_PROMPT)
    qa_chain_llm = LLMChain(llm=llm, prompt=QA_PROMPT)
    qa_chain = StuffDocumentsChain(llm_chain=qa_chain_llm, document_variable_name="context")

    last_user = history[-1].content.strip()
    lower_user = last_user.lower()
    history_text = "\n".join(m.content for m in history)

    # 1) Intent detection with heuristic for "what is / tell me about"
    if re.match(r"^(what is|tell me about)", lower_user):
        intent = "medical_fact"
    else:
        intent = classify_chain.run({"message": last_user, "history": history_text}).strip().lower()

    # 2) Vague symptom clarification
    if intent == "vague_symptom":
        clar_prompt = (
            "I’m sorry you’re not feeling well. "
            "Could you describe your symptoms in more detail—"
            "Use bullet points if needed."
            "for example, what you’re experiencing (fever, cough, headache), "
            "when it started, and how severe it feels?"
        )
        return {"response": clar_prompt, "source_tag": "clarification", "intent": intent}

    # 3) Topic extraction when needed
    topic = extract_chain.run({"message": last_user, "history": history_text}).strip().lower()
    if not topic or topic == "unknown":
        ask_topic = (
            "If you have a specific disease or topic in mind (e.g. malaria, flu), "
            "please let me know. Otherwise, feel free to keep describing how you feel, "
            "and I’ll help figure it out together."
        )
        return {"response": ask_topic, "source_tag": "clarification", "intent": intent}
    memory_store[f"{session_id}_topic"] = topic

    # 4) WHO retrieval for medical_fact or follow_up_detail
    if intent in ("medical_fact", "follow_up_detail"):
        if intent == "follow_up_detail" and f"{session_id}_last_docs" in memory_store:
            docs = memory_store[f"{session_id}_last_docs"]
        else:
            docs = retriever.get_relevant_documents(last_user)
        memory_store[f"{session_id}_last_docs"] = docs

        # Pick top-2 by similarity
        docs_sorted = sorted(
            docs,
            key=lambda d: util.cos_sim(
                similarity_model.encode(last_user, convert_to_tensor=True),
                similarity_model.encode(d.page_content, convert_to_tensor=True)
            ).item(),
            reverse=True
        )[:2]
        answer = qa_chain.run(input_documents=docs_sorted, question=last_user).strip()

        # Compute grounding metrics and append citations (as before)
        emb_ans = similarity_model.encode(answer, convert_to_tensor=True)
        emb_docs = similarity_model.encode([re.sub(r"\s+"," ",d.page_content) for d in docs_sorted], convert_to_tensor=True)
        sims = util.cos_sim(emb_ans, emb_docs)[0].cpu().tolist()
        max_sim = max(sims, default=0.0)
        avg_sim = sum(sims)/len(sims) if sims else 0.0

        # Citation presence
        unique_srcs = []
        for d in docs_sorted:
            src = d.metadata.get("source","")
            if src and src not in unique_srcs:
                unique_srcs.append(src)
        hits = sum(1 for src in unique_srcs if src.lower() in answer.lower())
        cit_score = (hits+1)/(len(unique_srcs)+1) if unique_srcs else 0.0

        # Overlap
        overlap = best_overlap_ratio(answer, [d.page_content for d in docs_sorted])

        # Combine
        grounding_score = max(0.0, min(1.0, 0.3 + 0.7*(0.4*max_sim+0.2*avg_sim+0.2*cit_score+0.2*overlap)))

        # Append sources
        if unique_srcs:
            if len(unique_srcs)==1:
                if unique_srcs[0] not in answer:
                    answer+=f"\n\n(See source: {unique_srcs[0]})"
            else:
                answer+="\n\n**Sources:**\n"+"\n".join(f"- {s}" for s in unique_srcs)

        return {"response": answer, "source_tag": "WHO", "intent": intent, "grounding_score": grounding_score
                }

    # 5) Real-time news lookup
    if intent == "real_time" and allow_search:
        tavily_search = get_tavily_search_tool()
        real_query = last_user if topic in lower_user else f"{topic} {last_user}"
        snippets = tavily_search(real_query)
        summary = summarize_results_with_llm(llm, snippets, real_query)
        resp = f"📰 **Latest on {topic.title()}:** {summary}"
        return {"response": resp, "source_tag": "internet", "intent": intent}

    # 6) Fallback chat
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    chat_chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)
    fallback = chat_chain.run(input=last_user)
    return {"response": fallback, "source_tag": "LLM", "intent": intent}
