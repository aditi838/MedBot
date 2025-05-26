import argparse
import re
import time
import pandas as pd
import requests
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util


# Metric helpers
def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def token_f1(pred, gold):
    pt = word_tokenize(normalize(pred))
    gt = word_tokenize(normalize(gold))
    if not pt or not gt:
        return 0.0
    common = set(pt) & set(gt)
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


sim_model = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_sim(pred, gold):
    v1 = sim_model.encode(pred, convert_to_tensor=True)
    v2 = sim_model.encode(gold, convert_to_tensor=True)
    return util.cos_sim(v1, v2).item()


# Evaluation runner
def main(csv_path, model_name, provider, host, port, allow_search):
    df = pd.read_csv(csv_path)
    results = []

    for idx, row in df.iterrows():
        question = row['question']
        gold     = row['answer']

        payload = {
            "session_id": f"eval_{idx}",
            "model_name": model_name,
            "model_provider": provider,
            "messages": [question],
            "allow_search": allow_search,
            "user_metadata": {"age": None, "gender": None}
        }

        url = f"http://{host}:{port}/chat"
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        resp = r.json()

        pred = resp.get("response", "")

        f1  = token_f1(pred, gold)
        sim = semantic_sim(pred, gold)

        results.append({
            "question": question,
            "gold_answer": gold,
            "pred_answer": pred,
            "token_f1": f1,
            "semantic_sim": sim
        })

        time.sleep(0.2)

    out = pd.DataFrame(results)
    print("\n=== AVERAGES ===")
    print(f"Token F1:    {out['token_f1'].mean():.3f}")
    print(f"Sem Sim:     {out['semantic_sim'].mean():.3f}")

    out.to_csv("eval_results_LLAMA_VERSATILE.csv", index=False)
    print("\nSaved detailed metrics to evaluation_results_api_new.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,     default="disease_qna_pairs.csv")
    parser.add_argument("--model", type=str,   required=True)
    parser.add_argument("--provider", type=str,default="Groq")
    parser.add_argument("--host", type=str,    default="127.0.0.1")
    parser.add_argument("--port", type=int,    default=9999)
    parser.add_argument("--allow-search", action="store_true")
    args = parser.parse_args()

    main(
        csv_path     = args.csv,
        model_name   = args.model,
        provider     = args.provider,
        host         = args.host,
        port         = args.port,
        allow_search = args.allow_search,
    )
