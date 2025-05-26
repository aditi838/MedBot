from fastapi import FastAPI, HTTPException
from statistics import mean
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from ai_agent import get_response_from_ai_agent, system_prompt
from evaluator import evaluate_chatbot


ALLOWED_MODELS = ["llama3-70b-8192", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"]

app = FastAPI(title="Healthcare AI Chatbot with Memory")


class UserMetadata(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None


class RequestState(BaseModel):
    session_id: str
    model_name: str
    model_provider: str
    messages: List[str]
    allow_search: bool
    user_metadata: Optional[UserMetadata] = None


@app.post("/chat")
async def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name.")

    # Build history
    history = []
    for i, msg in enumerate(request.messages):
        if i % 2 == 0:
            history.append(HumanMessage(content=msg))
        else:
            history.append(AIMessage(content=msg))

    # Append metadata
    if request.user_metadata and history:
        meta_parts = []
        if request.user_metadata.age is not None:
            meta_parts.append(f"Age={request.user_metadata.age}")
        if request.user_metadata.gender:
            meta_parts.append(f"Gender={request.user_metadata.gender}")
        if meta_parts:
            meta = f"Patient Info → {', '.join(meta_parts)}\n"
            for i in range(len(history) - 1, -1, -1):
                if isinstance(history[i], HumanMessage):
                    history[i] = HumanMessage(content=meta + history[i].content)
                    break

    result = get_response_from_ai_agent(
        session_id=request.session_id,
        llm_id=request.model_name,
        history=history,
        allow_search=request.allow_search,
        system_prompt=system_prompt,
        provider=request.model_provider
    )

    return {
        "response": result.get("response", "No response."),
        "source_tag": result.get("source_tag", ""),
        "grounding_score": result.get("grounding_score")
    }


@app.get("/evaluate")
def evaluate():
    results = evaluate_chatbot()
    return {
        "summary": {
            "average_time": round(mean([r['response_time'] for r in results]), 2),
            "average_accuracy": round(mean([r['accuracy'] for r in results]), 2),
            "hallucination_rate": round(sum(r['hallucination'] for r in results) / len(results), 2),
        },
        "details": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)
