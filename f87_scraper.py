import requests
from bs4 import BeautifulSoup
import time
import json
from tqdm import tqdm
import nltk
from transformers import GPT2TokenizerFast

# === CONFIGURATION ===
BASE_URL = "https://f87.bimmerpost.com/forums/"
FORUM_URL_TEMPLATES = [
    BASE_URL + "forumdisplay.php?f=652&page={}",  # Cosmetic
    BASE_URL + "forumdisplay.php?f=646&page={}",  # Wheels/Tires
    BASE_URL + "forumdisplay.php?f=653&page={}",  # Maintenance
    BASE_URL + "forumdisplay.php?f=654&page={}",  # Suspension and Brakes
    BASE_URL + "forumdisplay.php?f=722&page={}",  # S55
    BASE_URL + "forumdisplay.php?f=651&page={}",  # N55
    BASE_URL + "forumdisplay.php?f=660&page={}"   # Track
]
HEADERS = {
    "User-Agent": "F87M2-RAG-Bot (Contact: mattr832@gmail.com)"
}
DELAY = 1.0
MAX_PAGES = 40
THREADS_PER_PAGE = 36
MAX_TOKENS_PER_CHUNK = 400
OVERLAP_TOKENS = 50

# === FETCHING ===
def fetch_html(url):
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 429:
            print("Rate limited. Sleeping longer...")
            time.sleep(5)
            return fetch_html(url)
        else:
            print(f"Skipped {url} - Status: {response.status_code}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def extract_thread_links(forum_page_html):
    soup = BeautifulSoup(forum_page_html, "html.parser")
    links = []
    titles = []

    for td in soup.find_all("td", class_="alt1"):
        a_tag = td.find("a", id=lambda x: x and x.startswith("thread_title_"))
        if a_tag:
            href = a_tag.get("href")
            title = a_tag.get_text(strip=True)
            if href and "showthread.php" in href:
                full_url = BASE_URL + "/" + href.lstrip("/")
                links.append(full_url)
                titles.append(title)

    return list(zip(titles, links))

def extract_thread_content(thread_html):
    soup = BeautifulSoup(thread_html, "html.parser")

    # Remove quoted replies
    for quote in soup.find_all("div", class_=["quote", "bbcode_quote"]):
        quote.decompose()
    posts = soup.select("div.thePostItself")
    content = "\n\n".join(post.get_text(strip=True) for post in posts)
    return {"content": content}

def crawl_threads(start_page=1, max_pages=MAX_PAGES, threads_per_page=THREADS_PER_PAGE):
    all_threads = []

    for template in FORUM_URL_TEMPLATES:
        forum_id = template.split("forumdisplay.php?f=")[1].split("&")[0]
        print(f"\n🔍 Scraping section f={forum_id}...")

        for page_num in range(start_page, start_page + max_pages):
            page_url = template.format(page_num)
            html = fetch_html(page_url)
            if not html:
                continue

            title_link_pairs = extract_thread_links(html)[:threads_per_page]

            for title, thread_url in tqdm(title_link_pairs, desc=f"Page {page_num}"):
                thread_html = fetch_html(thread_url)
                if thread_html:
                    data = extract_thread_content(thread_html)
                    data["title"] = title
                    data["url"] = thread_url
                    data["forum_id"] = forum_id
                    all_threads.append(data)
                time.sleep(DELAY)

    return all_threads

# === CHUNKING ===
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def chunk_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, overlap=OVERLAP_TOKENS):
    if not text:
        return []

    tokens = tokenizer.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text.strip())

        # Move the window forward with overlap
        start += max_tokens - overlap

    return chunks

def chunk_all_threads(threads): 
    chunked_docs = []
    for thread in threads:
        try:
            chunks = chunk_text(thread.get("content", ""))
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "title": thread.get("title", "Untitled"),
                    "chunk_index": i,
                    "text": chunk,
                    "url": thread["url"],
                    "forum_id": thread.get("forum_id", "unknown")
                })
        except Exception as e:
            print(f"[Error] Skipping thread: '{thread.get('title', 'Untitled')}' - {e}")
            continue
    return chunked_docs

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚗 Starting multi-forum F87 M2 scrape and chunking...")
    threads = crawl_threads()
    with open("f87_threads.json", "w", encoding="utf-8", errors="ignore") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)

    chunked = chunk_all_threads(threads)
    with open("f87_chunks.json", "w", encoding="utf-8", errors="ignore") as f:
        json.dump(chunked, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Scraped {len(threads)} threads and created {len(chunked)} chunks.")
