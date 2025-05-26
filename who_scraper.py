# who_scraper.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain.docstore.document import Document


BASE_URL = "https://www.who.int"
FACTSHEET_INDEX = "https://www.who.int/news-room/fact-sheets"
CHROMA_STORE_DIR = "chroma_store"
BATCH_SIZE = 5000  # Set a safe batch size under 5461


def get_fact_sheet_links():
    print("🔍 Fetching WHO fact sheet URLs...")
    try:
        res = requests.get(FACTSHEET_INDEX)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/news-room/fact-sheets/detail/" in href:
                full_url = urljoin(BASE_URL, href.split("?")[0])
                if full_url not in links:
                    links.append(full_url)

        print(f"✅ Found {len(links)} fact sheet URLs.")
        return links
    except Exception as e:
        print(f"❌ Error fetching fact sheets: {e}")
        return []


def scrape_who_page(url):
    try:
        print(f"📰 Scraping: {url}")
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        main_content = soup.find("main")
        text = main_content.get_text(separator="\n", strip=True) if main_content else soup.get_text()
        return text.strip()
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return ""


def load_who_to_chroma():
    links = get_fact_sheet_links()
    if not links:
        print("⚠️ No links found. Exiting.")
        return

    texts = []
    metadatas = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for url in links:
        content = scrape_who_page(url)
        if not content:
            continue

        chunks = splitter.split_text(content)
        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        texts.extend(valid_chunks)
        metadatas.extend([{"source": url}] * len(valid_chunks))
        time.sleep(0.5)

    if not texts:
        print("⚠️ No valid text chunks. Nothing to insert into ChromaDB.")
        return

    print(f"🔢 Preparing {len(texts)} chunks for vector DB...")

    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_STORE_DIR, embedding_function=embedder)

    docs = [
        Document(page_content=chunk, metadata=meta)
        for chunk, meta in zip(texts, metadatas)
        if chunk.strip()
    ]

    print(f"✅ First sample:\n\n{docs[0].page_content[:500]}...\n")

    # Split into batches
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        vectordb.add_documents(batch)
        print(f"✅ Inserted batch {i // BATCH_SIZE + 1} with {len(batch)} documents.")

    vectordb.persist()
    print("✅ WHO Fact Sheets stored in ChromaDB.")


if __name__ == "__main__":
    load_who_to_chroma()
