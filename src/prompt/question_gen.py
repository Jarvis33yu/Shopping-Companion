single_product_question_prompt = """
# Task
Write a query to search for a product on the e-commerce platform.

# Product Information
{product_name}

# Important Notes
- Only use the basic product category name (no more than 5 tokens) from the title.
- Don't mention specific product attributes.
- Don't repeat the product title in the query.

Generate the query without any other text:
"""


add_on_deals_question_prompt = """
# Task
Write a query to search for a product bundle (include multiple products) on the e-commerce platform.

# Product Information
{product_names}

# Deal Information
Voucher Details: {voucher}
Budget: ${budget}

# Important Notes
- The query MUST include the voucher details and budget.
- Clearly state the multiple products in the bundle and use "," or "and" to connect them.
- The query should be like this: "Product bundle includes: X, Y, and Z. Voucher ... Budget ...".
- Only use the basic product category name (no more than 5 tokens) from the title.
- Don't mention specific product attributes.
- Don't repeat the product title in the query.

Generate the query without any other text:
"""


repeat_purchase_question_prompt = """
# Task
Generate a query that prompts the AI assistant to remind the customer to repurchase the product.

# Example
The query should be like this:
```plaintext
Given a list of products, retrieve the user's purchase history in memory and determine whether a repeat‑purchase reminder is needed.

Reminder criteria (both must be true for a product):
- The product is purchased regularly (the user has bought it multiple times).
- Days since the most recent purchase ≥ (repeat purchase cycle − 3 days).

Output:
- Find products that are purchased repeatedly and determine their repurchase cycles.
- Based on current date and those cycles, decide whether a repurchase reminder should be sent.

Product List:
1. product1
2. product2
```

# Product Information
{product_names}

# Important Notes
1. Include all products.
2. Include the reorder reminder criteria.
3. Output in human-friendly format.
4. Simplify the product name, but do not lose its uniqueness.
5. The person being questioned is the AI assistant, not the customer.

Generate the query without any other text:
"""


complement_question_prompt = """
# Task
Generate a query that prompts the AI assistant to retrieve the user's purchase history and recommend complementary products.

# Example
The query should be like this:
```plaintext
Identify the products the customer said they have purchased in the past week, then recommend complementary products based on those products.
```

# Important Notes
1. Explicitly mention "past week" or "last week".
2. Don't specify any product names.
3. Don't mention specific product attributes or features.
4. Keep it conversational and natural.
5. The person being questioned is the AI assistant, not the customer.

Generate the query without any other text:
"""
