#!/usr/bin/env python3
"""
Bitovi Blog Scraper with FAISS (Windows 7 Compatible)
Uses TF-IDF + FAISS for fast vector search without sentence-transformers
"""

import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    'requests': 'requests>=2.31.0',
    'bs4': 'beautifulsoup4>=4.12.0',
    'sklearn': 'scikit-learn>=1.0.0',
    'numpy': 'numpy>=1.21.0',
    'faiss': 'faiss-cpu>=1.7.2',
    'lxml': 'lxml>=4.9.0',
    'pickle': None,
    'json': None
}

def install_package(pip_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

def check_and_install_dependencies():
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module if module != 'bs4' else 'bs4')
        except ImportError:
            if pip_name:
                install_package(pip_name)

check_and_install_dependencies()

import os
import requests
import pickle
import numpy as np
import time
import uuid
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BitoviFAISSScraper")

class BitoviScraper:
    def __init__(self, base_url="https://www.bitovi.com/blog", db_dir="./db"):
        self.base_url = base_url
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)

        self.documents_path = os.path.join(db_dir, "documents.pkl")
        self.metadata_path = os.path.join(db_dir, "metadata.pkl")
        self.vectorizer_path = os.path.join(db_dir, "vectorizer.pkl")
        self.faiss_index_path = os.path.join(db_dir, "faiss.index")

        self.documents = []
        self.metadata = []
        self.vectorizer = None
        self.faiss_index = None

        self.load_existing()

    def load_existing(self):
        if os.path.exists(self.documents_path):
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        if os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        if os.path.exists(self.faiss_index_path):
            self.faiss_index = faiss.read_index(self.faiss_index_path)

    def save_all(self):
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        if self.faiss_index:
            faiss.write_index(self.faiss_index, self.faiss_index_path)

    def rebuild_index(self):
        if not self.documents:
            logger.warning("No documents to index.")
            return

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        matrix = self.vectorizer.fit_transform(self.documents)
        matrix_np = matrix.toarray().astype('float32')

        self.faiss_index = faiss.IndexFlatL2(matrix_np.shape[1])
        self.faiss_index.add(matrix_np)
        logger.info("FAISS index rebuilt with %d documents", len(self.documents))

    def fetch(self, url):
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            return BeautifulSoup(res.content, 'lxml')
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return 'bitovi.com' in parsed.netloc and '/blog/' in parsed.path and not parsed.path.rstrip('/') == '/blog'

    def extract_links(self, soup):
        links = set()
        for a in soup.select('a[href*="/blog/"]'):
            href = a.get('href')
            if href:
                url = urljoin(self.base_url, href)
                if self.is_valid_url(url):
                    links.add(url)
        return list(links)

    def extract_blog(self, url):
        soup = self.fetch(url)
        if not soup:
            return None
        title = soup.find('h1')
        content = soup.find('main') or soup.find('article')
        if not content:
            return None
        text = content.get_text(separator=' ', strip=True)
        return {
            'url': url,
            'title': title.get_text(strip=True) if title else "",
            'content': text,
            'scraped_at': datetime.now().isoformat()
        }

    def add_blog(self, blog):
        if blog['url'] in [m['url'] for m in self.metadata]:
            return False
        full_text = blog['title'] + "\n" + blog['content']
        self.documents.append(full_text)
        self.metadata.append({
            'id': str(uuid.uuid4()),
            'url': blog['url'],
            'title': blog['title'],
            'scraped_at': blog['scraped_at'],
            'length': len(blog['content'])
        })
        return True

    def scrape_all(self):
        soup = self.fetch(self.base_url)
        links = self.extract_links(soup)
        added = 0
        for url in links:
            blog = self.extract_blog(url)
            if blog and self.add_blog(blog):
                added += 1
                logger.info("Added: %s", blog['title'])
                time.sleep(1)
        if added > 0:
            self.rebuild_index()
        self.save_all()

    def search(self, query, k=5):
        if not self.faiss_index or not self.vectorizer:
            logger.warning("No FAISS index or vectorizer found")
            return []
        vec = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = self.faiss_index.search(vec, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                data = self.metadata[idx].copy()
                data['similarity_score'] = float(distances[0][i])
                data['rank'] = i + 1
                data['content_preview'] = self.documents[idx][:300] + '...'
                results.append(data)
        return results

    def search_all(self, query, threshold=0.1):
        if not self.faiss_index or not self.vectorizer:
            logger.warning("No FAISS index or vectorizer found")
            return []
        vec = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = self.faiss_index.search(vec, len(self.documents))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata) and distances[0][i] <= threshold:
                data = self.metadata[idx].copy()
                data['similarity_score'] = float(distances[0][i])
                data['rank'] = i + 1
                data['content_preview'] = self.documents[idx][:300] + '...'
                results.append(data)
        return results

    def answer_queries(self):
        questions = [
            "What is Bitovi's latest blog post about?",
            "Can you show me all Bitovi articles about AI?",
            "How many articles does Bitovi have about AI?",
            "What kind of tools does Bitovi recommend for E2E testing?"
        ]
        for q in questions:
            print(f"\n>>> Query: {q}")
            if "How many articles" in q:
                results = self.search_all(q, threshold=0.1)
                print(f"Found {len(results)} relevant articles.")
            else:
                results = self.search(q, k=10)
                if not results:
                    print("No relevant articles found.")
                else:
                    for r in results:
                        print(f"[Rank {r['rank']}] {r['title']}\nURL: {r['url']}\nScore: {r['similarity_score']:.4f}\nPreview: {r['content_preview']}\n")

# main() remains unchanged
# if __name__ == "__main__":
#     scraper = BitoviScraper()
#     logger.info("Loaded %d documents", len(scraper.documents))
#     # scraper.scrape_all()
#     # scraper.answer_queries()
