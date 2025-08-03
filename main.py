from fastapi import FastAPI, Query
from scraper import BitoviScraper
from fastapi.responses import JSONResponse

app = FastAPI()
scraper = BitoviScraper()

@app.get("/")
def root():
    return {"message": "Bitovi scraper API is up"}

@app.get("/search")
def search(q: str = Query(...), k: int = 5):
    results = scraper.search(q, k)
    return JSONResponse(content=results)

@app.get("/search-all")
def search_all(q: str, threshold: float = 0.1):
    results = scraper.search_all(q, threshold)
    return JSONResponse(content=results)

@app.post("/scrape")
def scrape():
    scraper.scrape_all()
    return {"message": "Scraping complete and FAISS index updated."}
