import requests
from bs4 import BeautifulSoup
import time
import json
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# --- Configuration ---
BASE_URL = "https://f87.bimmerpost.com/forums/"
FORUM_URL_TEMPLATE = BASE_URL + "forumdisplay.php?f=653&page={}"
HEADERS = {
    "User-Agent": "F87M2-RAG-Bot (Contact: mattr832@gmail.com)"
}
DELAY = 1.0  # seconds between requests
MAX_PAGES = 1
THREADS_PER_PAGE = 36
MAX_TOKENS_PER_CHUNK = 400
OVERLAP_TOKENS = 50

# --- Utilities ---
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

    # Look for each thread title cell
    for td in soup.find_all("td", class_="alt1"):
        a_tag = td.find("a", id=lambda x: x and x.startswith("thread_title_"))
        if a_tag:
            href = a_tag.get("href")
            title = a_tag.get_text(strip=True)
            if href and "showthread.php" in href:
                full_url = BASE_URL + "/" + href.lstrip("/")  # ensure proper formatting
                links.append(full_url)
                titles.append(title)
    return list(zip(titles, links))

def extract_thread_content(thread_html):
    soup = BeautifulSoup(thread_html, "html.parser")

    # Get the thread title from the first <strong> inside a .smallfont div
    title_tag = soup.find("div", class_="smallfont")
    title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

    # Get all post content blocks (initial post + replies)
    post_divs = soup.find_all("div", class_="thePostItself")
    content = "\n\n".join(div.get_text(strip=True) for div in post_divs)

    return {
        "title": title,
        "content": content
    }

def crawl_threads(start_page=1, max_pages=MAX_PAGES, threads_per_page=THREADS_PER_PAGE):
    all_threads = []
    for page_num in range(start_page, start_page + max_pages):
        page_url = FORUM_URL_TEMPLATE.format(page_num)
        html = fetch_html(page_url)
        if not html:
            continue
        title_link_pairs = extract_thread_links(html)[:threads_per_page]

        for title, thread_url in tqdm(title_link_pairs, desc=f"Page {page_num}"):
            thread_html = fetch_html(thread_url)
            if thread_html:
                data = extract_thread_content(thread_html)
                data["title"] = title  # use title from the listing page
                data["url"] = thread_url
                all_threads.append(data)
            time.sleep(DELAY)
    return all_threads

# --- Chunking ---
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def chunk_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, overlap=OVERLAP_TOKENS):
    if not text:
        return []

    # Truncate very long text to avoid GPT2 tokenizer issues
    tokens = tokenizer.encode(str(text))[:1024]
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        start += max_tokens - overlap
    return chunks

def chunk_all_threads(threads):
    chunked_docs = []
    for thread in threads:
        chunks = chunk_text(thread.get("content", ""))
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "title": thread.get("title", "Untitled"),
                "chunk_index": i,
                "text": chunk,
                "url": thread["url"]
            })
    return chunked_docs

# --- Execution ---
if __name__ == "__main__":
    print("Starting Bimmerpost F87 M2 General Discussion scrape with chunking...")
    threads = crawl_threads()
    with open("f87_threads.json", "w", encoding="utf-8", errors="ignore") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)

    chunked = chunk_all_threads(threads)
    with open("f87_chunks.json", "w", encoding="utf-8", errors="ignore") as f:
        json.dump(chunked, f, indent=2, ensure_ascii=False)

    print(f"Done! Scraped {len(threads)} threads and created {len(chunked)} chunks.")
