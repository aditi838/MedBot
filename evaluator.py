import time
import pandas as pd
from langchain_core.messages import HumanMessage
from ai_agent import get_response_from_ai_agent, system_prompt
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model once
# Configurable model and thresholds
EVAL_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EVAL_HALLUCINATION_THRESHOLD = 0.6

# Load evaluation-specific embedding model
embed_model = SentenceTransformer(EVAL_EMBED_MODEL_NAME)


# ------------------ Visualization ------------------
def visualize_results(results):
    df = pd.DataFrame(results)

    # Accuracy by intent
    plt.figure(figsize=(6, 4))
    sns.barplot(x="expected_intent", y="accuracy", data=df)
    plt.title("Accuracy by Expected Intent")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy Score")
    plt.xlabel("Expected Intent")
    plt.tight_layout()
    plt.show()

    # Similarity score distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["similarity_score"], kde=True, bins=10, color='skyblue')
    plt.title("Similarity Score Distribution (Model Self-Reported)")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Semantic similarity distribution
    if "semantic_similarity" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df["semantic_similarity"], kde=True, bins=10, color='mediumseagreen')
        plt.title("Semantic Similarity Distribution (Sentence Transformers)")
        plt.xlabel("Semantic Similarity Score")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # Hallucination count
    plt.figure(figsize=(4, 4))
    sns.countplot(x="hallucination", data=df)
    plt.title("Hallucination Count")
    plt.xlabel("Hallucination")
    plt.ylabel("Number of Responses")
    plt.tight_layout()
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(df["expected_intent"], df["predicted_intent"], labels=df["expected_intent"].unique())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df["expected_intent"].unique())
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Intent Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ------------------ Scoring ------------------
def keyword_overlap_score(answer, expected_keywords):
    return sum(1 for word in expected_keywords if word.lower() in answer.lower()) / len(expected_keywords) if expected_keywords else 0.0

def clarification_score(answer):
    triggers = ["could you", "please describe", "more detail", "what kind", "symptoms", "feel", "how long"]
    return 1.0 if any(t in answer.lower() for t in triggers) else 0.0

def semantic_similarity_score(reference: str, hypothesis: str) -> float:
    ref_embedding = embed_model.encode(reference, convert_to_tensor=True)
    hyp_embedding = embed_model.encode(hypothesis, convert_to_tensor=True)
    similarity = util.cos_sim(ref_embedding, hyp_embedding)
    return round(float(similarity), 3)


def calculate_classification_metrics(expected, predicted, label_list=None):
    precision = precision_score(expected, predicted, average='weighted', labels=label_list, zero_division=0)
    recall = recall_score(expected, predicted, average='weighted', labels=label_list, zero_division=0)
    f1 = f1_score(expected, predicted, average='weighted', labels=label_list, zero_division=0)
    accuracy = accuracy_score(expected, predicted)
    return round(precision, 3), round(recall, 3), round(f1, 3), round(accuracy, 3)


# ------------------ Evaluation ------------------
def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return []

    df = pd.read_csv(csv_path)

    if df.empty:
        print("Dataset is empty.")
        return []

    if "expected_keywords" not in df.columns or "question" not in df.columns:
        print("Missing required columns in dataset.")
        return []

    df["expected_keywords"] = df["expected_keywords"].apply(lambda x: x.split(",") if isinstance(x, str) else [])
    return df.to_dict(orient="records")


def evaluate_chatbot():
    results = []
    dataset = load_dataset("sample_eval_dataset.csv")
    if not dataset:
        print("No data to evaluate.")
        return []

    session_id = "test_session"
    model_name = "llama3-70b-8192"
    provider = "Groq"
    print(f"Loaded {len(dataset)} samples from dataset.")

    for idx, sample in enumerate(dataset):
        query = sample["question"]
        expected_keywords = sample.get("expected_keywords", [])
        expected_intent = sample.get("intent", "unknown")
        expected_answer = sample.get("expected_answer", "")
        print(f"\n[{idx+1}/{len(dataset)}] Evaluating: {query}")

        try:
            history = [HumanMessage(content=query)]
            start_time = time.time()

            response = get_response_from_ai_agent(
                session_id=session_id,
                llm_id=model_name,
                history=history,
                allow_search=True,
                system_prompt=system_prompt,
                provider=provider
            )

            end_time = time.time()
            output = response.get("response", "")
            source = response.get("source_tag", "")
            similarity = response.get("similarity_score", 1.0)
            predicted_intent = response.get("intent", "unknown")
            duration = round(end_time - start_time, 2)

            if not output:
                print("⚠️ Empty response received.")
                continue

            if expected_intent == "vague_symptom":
                accuracy = clarification_score(output)
            else:
                accuracy = keyword_overlap_score(output, expected_keywords)

            hallucination = similarity < EVAL_HALLUCINATION_THRESHOLD if expected_intent != "vague_symptom" else False
            semsim = semantic_similarity_score(expected_answer, output)

            result = {
                "question": query,
                "response": output,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "accuracy": round(accuracy, 2),
                "response_time": duration,
                "similarity_score": round(similarity, 2),
                "hallucination": hallucination,
                "source_tag": source,
                "semantic_similarity": semsim
            }

            results.append(result)

            # Save intermediate results
            pd.DataFrame(results).to_csv("debug_results.csv", index=False)

        except Exception as e:
            print(f"❌ Error processing question: {e}")
    # Compute classification metrics
    expected_intents = [r["expected_intent"] for r in results]
    predicted_intents = [r["predicted_intent"] for r in results]
    label_list = list(set(expected_intents + predicted_intents))

    precision, recall, f1, acc = calculate_classification_metrics(expected_intents, predicted_intents, label_list)
    print(f"\n🔍 Intent Classification Metrics:")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1 Score:  {f1}")
    print(f"Accuracy:  {acc}")

    return results


def export_to_csv(results, filename="final_eval_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"✅ Results exported to {filename}")


# ------------------ Main ------------------
if __name__ == "__main__":
    results = evaluate_chatbot()
    if results:
        pprint(results)
        visualize_results(results)
        export_to_csv(results)
    else:
        print("❌ No results to display or export.")