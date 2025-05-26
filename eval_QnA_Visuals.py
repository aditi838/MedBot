# viz_eval_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load evaluation results
FILE = "evaluation_results_api_new.csv"
assert os.path.exists(FILE), f"{FILE} not found. Run the evaluation script first."

df = pd.read_csv(FILE)

# Create output directory for plots
os.makedirs("eval_plots", exist_ok=True)

# --- 1. Histogram of Token F1 ---
plt.figure(figsize=(8, 5))
sns.histplot(df['token_f1'], bins=20, kde=True, color='skyblue')
plt.title("Token F1 Distribution")
plt.xlabel("Token F1 Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eval_plots/token_f1_distribution.png")
plt.close()

# --- 2. Histogram of Semantic Similarity ---
plt.figure(figsize=(8, 5))
sns.histplot(df['semantic_sim'], bins=20, kde=True, color='salmon')
plt.title("Semantic Similarity Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eval_plots/semantic_similarity_distribution.png")
plt.close()

# --- 3. Scatter plot of F1 vs Semantic Similarity ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x="token_f1", y="semantic_sim", data=df)
plt.title("Token F1 vs Semantic Similarity")
plt.xlabel("Token F1")
plt.ylabel("Semantic Similarity")
plt.grid(True)
plt.tight_layout()
plt.savefig("eval_plots/f1_vs_semantic_sim.png")
plt.close()

# --- 4. Correlation Heatmap ---
plt.figure(figsize=(6, 4))
sns.heatmap(df[['token_f1', 'semantic_sim']].corr(), annot=True, cmap='coolwarm')
plt.title("Metric Correlation Heatmap")
plt.tight_layout()
plt.savefig("eval_plots/metric_correlation_heatmap.png")
plt.close()

# --- 5. Save low performance cases ---
low_perf = df[(df['token_f1'] < 0.4) & (df['semantic_sim'] < 0.6)]
low_perf.to_csv("low_performance_cases.csv", index=False)
print(f"Saved {len(low_perf)} low-performance cases to 'low_performance_cases.csv'")

print("✅ Visualizations saved in 'eval_plots/' and analysis complete.")
