from utils import remove_quoted_replies
from f87_scraper import chunk_all_threads
import json

# Load threads
with open("f87_threads.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

# Clean each thread's content
for thread in threads:
    if isinstance(thread.get("content"), str):
        thread["content"] = remove_quoted_replies(thread["content"])

chunked = chunk_all_threads(threads)
with open("f87_chunks.json", "w", encoding="utf-8", errors="ignore") as f:
    json.dump(chunked, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Created {len(chunked)} chunks.")

# # Save cleaned threads
# with open("f87_threads_cleaned.json", "w", encoding="utf-8") as f:
#     json.dump(threads, f, indent=2, ensure_ascii=False)
