import json
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt

# Load chunks
with open("f87_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Count tokens in each chunk
token_counts = [len(tokenizer.encode(chunk["text"])) for chunk in chunks]

# Stats
avg_tokens = sum(token_counts) / len(token_counts)
print(f"Average tokens per chunk: {avg_tokens:.2f}")
print(f"Max tokens: {max(token_counts)}")
print(f"Min tokens: {min(token_counts)}")
print(f"Total chunks: {len(token_counts)}")

# Plot histogram
plt.hist(token_counts, bins=30, edgecolor="black")
plt.title("Token Count Distribution per Chunk")
plt.xlabel("Tokens")
plt.ylabel("Number of Chunks")
plt.axvline(avg_tokens, color='red', linestyle='dashed', label=f"Avg = {avg_tokens:.1f}")
plt.legend()
plt.tight_layout()
plt.show()
