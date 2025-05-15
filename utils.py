import json
import re

def remove_quoted_replies(text):
    # This pattern matches 'Quote:Originally Posted by...' and all the text that follows until the next quote or newline.
    return re.sub(r'Quote:Originally Posted by.*?(?=(\n\n|Quote:|$))', '', text, flags=re.DOTALL)

def split_threads(threads):
    with open("f87_threads.json", "r", encoding="utf-8") as f:
        threads = json.load(f)

    chunk_size = 2000
    for i in range(0, len(threads), chunk_size):
        part = threads[i:i + chunk_size]
        with open(f"f87_threads_part_{i // chunk_size + 1}.json", "w", encoding="utf-8") as f_out:
            json.dump(part, f_out, indent=2, ensure_ascii=False)
