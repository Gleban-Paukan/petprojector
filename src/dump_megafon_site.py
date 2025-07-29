import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json

root_url = "https://moscow.shop.megafon.ru/"
max_depth = 5

root_netloc = urlparse(root_url).netloc
parts = root_netloc.split(".")
base_domain = ".".join(parts[-2:])  # "megafon.ru"

visited_global = set()
chains = []

def crawl(url, path, depth):
    if depth > max_depth:
        return

    if depth > 0:
        chains.append(path.copy())

    if depth == max_depth:
        return

    if url in visited_global:
        return
    visited_global.add(url)

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        return

    soup = BeautifulSoup(resp.text, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = urljoin(root_url, a['href'])
        parsed = urlparse(href)

        host = parsed.netloc.lower()
        if not (host == base_domain or host.endswith("." + base_domain)):
            continue

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean_url in path:
            continue

        path.append(clean_url)
        print(f"{'  '*depth}-> {clean_url}")
        crawl(clean_url, path, depth + 1)
        path.pop()

if __name__ == "__main__":
    crawl(root_url, [root_url], 0)

    print(f"\nVisited {len(visited_global)} unique URLs.")
    with open("chains.json", "w", encoding="utf-8") as f:
        json.dump(chains, f, ensure_ascii=False, indent=2)
    print("Chains saved to chains.json")
