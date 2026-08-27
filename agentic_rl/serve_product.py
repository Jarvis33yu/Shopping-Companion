# File: src/serve_product_fastapi.py

import argparse
import multiprocessing
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import ujson as json
from pyserini.search.lucene import LuceneSearcher
from fastapi import FastAPI, Query, HTTPException

# ================================
# Global Variables (set on startup)
# ================================

app = FastAPI(
    title="Product Search API",
    description="FastAPI version of product search with Lucene backend",
    version="1.0"
)

searcher: Optional[LuceneSearcher] = None
thread_pool: Optional[ThreadPoolExecutor] = None


# ===================================
# Core Logic (unchanged from original)
# ===================================

def _process_hit(
    hit,
    shop_id: Optional[str],
    price_min: Optional[float],
    price_max: Optional[float]
) -> Optional[Dict]:
    """Process a single hit and return product data if it matches filters."""
    try:
        doc = searcher.doc(hit.docid)
        if not doc:
            return None
        product = json.loads(doc.raw())["product"]

        seller_id = product["seller_id"]
        product_price = product["price"]

        if shop_id and shop_id != seller_id:
            return None
        if price_min is not None and (product_price < price_min or product_price > price_max):
            return None

        return {
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "shop_id": seller_id,
            "price": product_price,
        }
    except Exception as e:
        print(f"Error processing hit {hit.docid}: {e}", file=sys.stderr)
        return None


def _product_search(
    query: str, k: int, shop_id: Optional[str] = None, price: Optional[str] = None
) -> List[Dict]:
    # Parse price range
    price_min = price_max = None
    if price:
        try:
            parts = price.split("-")
            low = parts[0].strip()
            high = parts[1].strip() if len(parts) > 1 else ""
            price_min = float(low) if low else 0.0
            price_max = float(high) if high else float("inf")
            assert price_min <= price_max
            assert price_min >= 0
        except Exception as e:
            print(f"Invalid price format '{price}': {e}", file=sys.stderr)
            return []

    capacity = k if not shop_id and not price else 10000
    hits = searcher.search(q=query, k=capacity, remove_dups=True)

    results = []

    # Fast path: no filter → sequential
    if not shop_id and price_min is None:
        for hit in hits:
            result = _process_hit(hit, None, None, None)
            if result:
                results.append(result)
                if len(results) >= k:
                    break
        return results

    # With filters → use thread pool
    futures_map = {}
    for idx, hit in enumerate(hits):
        future = thread_pool.submit(_process_hit, hit, shop_id, price_min, price_max)
        futures_map[future] = idx

    completed_results = {}
    for future in as_completed(futures_map):
        idx = futures_map[future]
        result = future.result()
        if result:
            completed_results[idx] = result
        if len(completed_results) >= k:
            break  # Early stop

    for idx in sorted(completed_results.keys()):
        results.append(completed_results[idx])
        if len(results) >= k:
            break

    return results


def _fetch_product_details(product_id: str) -> Optional[Dict]:
    """Fetch detailed product information by ID."""
    try:
        doc = searcher.doc(product_id)
        if not doc:
            return None
        product = json.loads(doc.raw())["product"]
        return {
            "product_id": product["product_id"],
            "attributes": product.get("attributes"),
            "options": product.get("options"),
        }
    except Exception as e:
        print(f"Error fetching product {product_id}: {e}", file=sys.stderr)
        return None


def _product_view(product_ids: List[str]) -> List[Dict]:
    if not product_ids:
        return []

    futures_map = {}
    for idx, pid in enumerate(product_ids):
        future = thread_pool.submit(_fetch_product_details, pid)
        futures_map[future] = idx

    completed_results = {}
    for future in as_completed(futures_map):
        idx = futures_map[future]
        result = future.result()
        if result:
            completed_results[idx] = result

    results = []
    for idx in sorted(completed_results.keys()):
        results.append(completed_results[idx])

    return results


# ====================
# API Routes
# ====================

@app.get("/")
def index():
    return {
        "usage": {
            "/product_search": "query, k=50, shop_id, price=min-max",
            "/product_view": "product_ids=id1,id2,..."
        }
    }


@app.get("/product_search")
def api_product_search(
    query: str = Query(..., description="Search query"),
    k: int = Query(50, ge=1, le=100, description="Number of results"),
    shop_id: Optional[str] = Query(None, description="Filter by shop ID"),
    price: Optional[str] = Query(None, description="Price range like '0-100'"),
):
    try:
        result = _product_search(query=query, k=k, shop_id=shop_id, price=price)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/product_view")
def api_product_view(
    product_ids: str = Query(..., description="Comma-separated product IDs"),
):
    try:
        ids = [pid.strip() for pid in product_ids.split(",") if pid.strip()]
        result = _product_view(ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Product view failed: {str(e)}")


# ==============================
# Startup Hook (called by uvicorn)
# ==============================

@app.on_event("startup")
def load_index_and_threadpool():
    import os
    global searcher, thread_pool

    # 设置默认值或从环境变量读取
    index_dir = os.getenv("INDEX_DIR", "data/product_indexes")
    worker_threads = int(os.getenv("WORKER_THREADS", multiprocessing.cpu_count() * 2))

    print(f"🚀 Loading Lucene index from: {index_dir}")
    searcher = LuceneSearcher(index_dir)

    print(f"🧵 Initializing thread pool with {worker_threads} workers")
    thread_pool = ThreadPoolExecutor(max_workers=worker_threads)

    print("✅ Product search service ready!")


@app.on_event("shutdown")
def shutdown_event():
    global thread_pool
    if thread_pool:
        thread_pool.shutdown(wait=True)
    print("👋 Product search service shutdown complete.")

# nohup uvicorn agentic_rl.serve_product:app --host 0.0.0.0 --port 5631 --workers 4 --limit-concurrency 1000 --timeout-keep-alive 30 --log-level info > logs/serve_product 2>&1 &
