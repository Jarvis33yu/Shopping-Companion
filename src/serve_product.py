import sys
import argparse
import multiprocessing
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import ujson as json
from pyserini.search.lucene import LuceneSearcher
from flask import Flask, request, jsonify
from waitress import serve


app = Flask(__name__)
searcher = None
thread_pool = None


def _process_hit(
    hit, shop_id: Optional[str], price_min: Optional[float], price_max: Optional[float]
) -> Optional[Dict]:
    """Process a single hit and return product data if it matches filters."""
    try:
        product = searcher.doc(hit.docid)
        if not product:
            return None
        product = json.loads(product.raw())["product"]

        # Cache frequently accessed fields to local variables
        seller_id = product["seller_id"]
        product_price = product["price"]

        # Check simpler filter first (string comparison)
        if shop_id and shop_id != seller_id:
            return None
        # Check price range (numeric comparison)
        if price_min is not None and (
            product_price < price_min or product_price > price_max
        ):
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
    # Parse price range once outside the loop
    price_min = price_max = None
    if price:
        try:
            price_range = price.split("-")
            price_min = float(price_range[0]) if price_range[0] else 0.0
            price_max = float(price_range[1]) if price_range[1] else float("inf")
            assert price_min <= price_max
            assert price_min >= 0
        except Exception as e:
            print(f"Error occurred during parsing price range: {e}", file=sys.stderr)
            return []

    capacity = k if not shop_id and not price else 10000
    hits = searcher.search(q=query, k=capacity, remove_dups=True)

    results = []

    # Fast path: no filters, process sequentially (avoid thread overhead)
    if not shop_id and price_min is None:
        for hit in hits:
            result = _process_hit(hit, None, None, None)
            if result is not None:
                results.append(result)
                if len(results) >= k:
                    break
        return results

    # With filters: use thread pool to process hits in parallel
    futures_map = {}

    # Submit all hits for parallel processing
    for idx, hit in enumerate(hits):
        future = thread_pool.submit(_process_hit, hit, shop_id, price_min, price_max)
        futures_map[future] = idx

    # Collect results as they complete, maintaining order
    completed_results = {}
    for future in as_completed(futures_map):
        idx = futures_map[future]
        result = future.result()
        if result is not None:
            completed_results[idx] = result

    # Sort by original hit order (relevance order) and take top k
    for idx in sorted(completed_results.keys()):
        results.append(completed_results[idx])
        if len(results) >= k:
            break

    return results


def _fetch_product_details(product_id: str) -> Optional[Dict]:
    """Fetch detailed product information by product ID."""
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

    # Use thread pool to fetch products in parallel
    results = []
    futures_map = {}

    # Submit all product IDs for parallel fetching
    for idx, product_id in enumerate(product_ids):
        future = thread_pool.submit(_fetch_product_details, product_id)
        futures_map[future] = idx

    # Collect results maintaining order
    completed_results = {}
    for future in as_completed(futures_map):
        idx = futures_map[future]
        result = future.result()
        if result is not None:
            completed_results[idx] = result

    # Sort by original order
    for idx in sorted(completed_results.keys()):
        results.append(completed_results[idx])

    return results


@app.route("/")
def index():
    usage = {
        "/product_search": "query,shop_id,price",
        "/product_view": "product_ids",
    }
    return jsonify(usage)


@app.route("/product_search")
def product_search():
    result = _product_search(
        query=request.args.get("query"),
        k=int(request.args.get("k")),
        shop_id=request.args.get("shop_id"),
        price=request.args.get("price"),
    )
    return jsonify(result)


@app.route("/product_view")
def product_view():
    result = _product_view(product_ids=request.args.get("product_ids").split(","))
    return jsonify(result)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--index_dir", type=str, required=True)
    args.add_argument("--host", type=str, default="0.0.0.0")
    args.add_argument("--port", type=int, default=5631)
    args.add_argument(
        "--worker_threads",
        type=int,
        default=None,
        help="Number of worker threads for parallel processing (default: CPU count * 2)",
    )
    args = args.parse_args()

    searcher = LuceneSearcher(args.index_dir)

    cores = multiprocessing.cpu_count()

    # Initialize thread pool for parallel document processing
    worker_threads = args.worker_threads or cores * 2
    thread_pool = ThreadPoolExecutor(max_workers=worker_threads)

    serve(
        app,
        host=args.host,
        port=args.port,
        threads=cores,
        expose_tracebacks=True,
        channel_timeout=10,
        cleanup_interval=10,
    )
