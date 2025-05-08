import requests
from bs4 import BeautifulSoup
from f87_scraper import *

main_url = 'https://f87.bimmerpost.com/forums/forumdisplay.php?f=653'
BASE_URL = "https://f87.bimmerpost.com/forums"

response = requests.get(main_url)
# print(response)
# print(response.text)

soup = BeautifulSoup(response.text, "html.parser")

# with open('saved_page.html', 'w', encoding='utf-8') as f:
#     f.write(soup.prettify())  # or just: f.write(str(soup)) to keep original formatting

# soup = BeautifulSoup(forum_page_html, "html.parser")
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

# print(titles)
# print(links)

thread_url = links[5]
thread_html = requests.get(thread_url)
thread_soup = BeautifulSoup(thread_html.text, "html.parser")

# with open('saved_thread.html', 'w', encoding='utf-8') as f:
#     f.write(thread_soup.prettify())  # or just: f.write(str(soup)) to keep original formatting

# Get the thread title from the first <strong> inside a .smallfont div
title_tag = thread_soup.find("div", class_="smallfont")
title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

# Get all post content blocks (initial post + replies)
post_divs = thread_soup.find_all("div", class_="thePostItself")
content = "\n\n".join(div.get_text(strip=True) for div in post_divs)

print(content)