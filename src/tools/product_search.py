import os
import sys
import time
from urllib.parse import quote_plus

import requests

from tools.base import BaseTool


PRODUCT_SEARCH_BASE_URL = os.getenv("PRODUCT_SEARCH_BASE_URL", "http://127.0.0.1:5631")
TIMEOUT = int(os.getenv("PRODUCT_SEARCH_TIMEOUT", 60))
MAX_RETRIES = int(os.getenv("PRODUCT_SEARCH_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("PRODUCT_SEARCH_RETRY_DELAY", 1.0))


class ProductSearch(BaseTool):
    name: str = "product_search"
    description: str = "Given a query and search for up to 50 relevant products. Optionally filter results by shop ID or price range."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords or phrase describing the desired product.",
            },
            "shop_id": {
                "type": "string",
                "description": "Restrict search to a specific shop using its unique Shop ID.",
            },
            "price": {
                "type": "string",
                "description": 'Filter products by price range in the format "min-max" (e.g., "0-100") or "min-" for no upper bound (e.g., "1000-").',
            },
        },
        "required": ["query"],
    }

    def execute(self, **kwargs):
        query = kwargs.get("query")
        shop_id = kwargs.get("shop_id")
        price = kwargs.get("price")

        if not query:
            return "No query provided."

        if price:
            try:
                price_range = price.split("-")
                price_range = [float(p) if p else float('inf') for p in price_range]
                assert price_range[0] <= price_range[1]
                assert price_range[0] >= 0
            except Exception as e:
                print(f"Error occurred during parsing price range: {e}", file=sys.stderr)
                return "Invalid price range. Please provide a price range in the format of 'min-max' (e.g., '0-100') or 'min-' for no upper bound (e.g., '1000-')."

        shop_id_str = ""
        if shop_id:
            shop_id_str = f"&shop_id={shop_id}"

        price_str = ""
        if price:
            price_str = f"&price={price}"

        url = f"{PRODUCT_SEARCH_BASE_URL}/product_search?query={quote_plus(query)}&k=50{shop_id_str}{price_str}"
        for i in range(MAX_RETRIES):
            try:
                resp = requests.get(url, timeout=TIMEOUT)
                resp.raise_for_status()
                results = resp.json()

                products = []
                for idx, product in enumerate(results):
                    product_id = product["product_id"]
                    product_name = product["product_name"]
                    shop_id = product["shop_id"]
                    price = product["price"]

                    products.append(f"{idx+1}. {product_name}\nProduct ID: {product_id}\nShop ID: {shop_id}\nPrice: ${price}")

                if products:
                    tool_response = f'A product search for "{query}" found {len(products)} results:\n\n' + "\n\n".join(products)
                    break
                else:
                    tool_response = f'Product search for "{query}" found no results.'
            except Exception as e:
                print(f"Error occurred during product search: {e}, retry {i + 1}/{MAX_RETRIES}", file=sys.stderr)
                tool_response = f"Error occurred during product search: {e}"
                time.sleep(RETRY_DELAY)
        return tool_response
