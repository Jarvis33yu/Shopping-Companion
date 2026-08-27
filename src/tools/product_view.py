import os
import sys
import time
from typing import List

import requests

from tools.base import BaseTool


PRODUCT_VIEW_BASE_URL = os.getenv("PRODUCT_VIEW_BASE_URL", "http://127.0.0.1:5631")
TIMEOUT = int(os.getenv("PRODUCT_VIEW_TIMEOUT", 60))
MAX_RETRIES = int(os.getenv("PRODUCT_VIEW_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("PRODUCT_VIEW_RETRY_DELAY", 1.0))


def fetch_product_detail(product_ids: List[str]) -> List[str]:
    if not product_ids:
        return []

    results = []
    for i in range(MAX_RETRIES):
        try:
            url = f"{PRODUCT_VIEW_BASE_URL}/product_view?product_ids={','.join(product_ids)}"
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            results = resp.json()

            d = {p["product_id"]: p for p in results}
 
            tool_responses = []
            for product_id in product_ids:
                if product_id in d:
                    product = d[product_id]
                    attributes = product.get("attributes")
                    options = product.get("options")

                    attributes_str = ""
                    if attributes:
                        attributes_str = "; ".join([f"{k} = {', '.join(vs)}" for k, vs in attributes.items()])

                    options_str = ""
                    if options:
                        for option in options:
                            if option:
                                options_str += "- " +"; ".join([f"{k} = {', '.join(vs)}" for k, vs in option.items()]) + "\n"

                    tool_responses.append(f"The attributes and options of product {product_id} are as follows:\nAttributes: {attributes_str}\nOptions:\n{options_str}".strip())
                else:
                    tool_responses.append(f"The attributes and options of product {product_id} are not found.")
            break
        except Exception as e:
            print(
                f"Error occurred during fetching product attributes and options: {e}, retry {i + 1}/{MAX_RETRIES}",
                file=sys.stderr,
            )
            tool_responses = ["Error occurred during fetching product attributes and options: {e}"]
            time.sleep(RETRY_DELAY)
    return tool_responses


class ProductView(BaseTool):
    name: str = "product_view"
    description: str = "Given an array of product IDs, retrieve the product attributes and options for each product."
    parameters: dict = {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string", "description": "The product ID."},
                "minItems": 1,
                "description": "An array of product IDs.",
            },
        },
        "required": ["product_ids"],
    }

    def execute(self, **kwargs):
        product_ids = kwargs.get("product_ids")

        if not product_ids:
            return "No product IDs provided."
        if not isinstance(product_ids, list):
            return "Product IDs must be a list of strings."

        tool_responses = fetch_product_detail(product_ids)
        delimiter = "\n\n" + "=" * 10 + "\n\n"
        return delimiter.join(tool_responses).strip()
