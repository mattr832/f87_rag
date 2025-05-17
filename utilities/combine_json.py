import json

# Load the first JSON file
with open("f87_threads1.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

# Load the second JSON file
with open("f87_threads2.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

# Combine both lists
combined = data1 + data2

# Optional: remove duplicates by URL or title if needed
seen_urls = set()
unique_combined = []
for item in combined:
    if item["url"] not in seen_urls:
        unique_combined.append(item)
        seen_urls.add(item["url"])

# Save the combined result
with open("f87_threads.json", "w", encoding="utf-8") as f_out:
    json.dump(combined, f_out, indent=2, ensure_ascii=False)

print("✅ Combined data saved to combined_threads.json")
