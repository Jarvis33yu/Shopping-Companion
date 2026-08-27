single_product_stage_1_evaluation_prompt = """
You are an expert evaluator for large language model outputs. Your task is to assess a ReAct-style assistant's ability to retrieve user memories and extract user preferences accurately.

## Evaluation Task

Please evaluate the model's output based on the following information:

**User Query:**
{question}

**Model Output (answer):**
{answer}

**Ground Truth (Reference Information):**
- Product Name: {product_name}
- User Expected Features/Preferences: 
{wanted_features}

## Evaluation Dimensions

Please score the model's output on the following dimensions:

### 1. Query Relevance
- Is the model's answer relevant to the user query?
- Does the answer directly address the question asked?

**Scoring Criteria:**
- 1: Answer is relevant and directly addresses the user query
- 0: Answer is irrelevant or does not address the user query

### 2. Preference Match Count
- How many of the User Expected Features/Preferences are mentioned or matched in the model's answer?
- Count and return only the number of matched features.

## Output Format

Please provide only the evaluation scores in the following JSON format:

```json
{{
  "query_relevance": <0 or 1>,
  "preference_match_count": <matched_count>
}}
```
"""

single_product_stage_2_evaluation_prompt = """You are an expert product-query matching evaluator.

## Task
Evaluate if a recommended product matches a user query and their wanted features.

**User Query:**
{question}

**User Preferences (Key Desired Features):**  
These are the specific attributes or functionalities the user cares about—directly related to their search intent. 
The product should ideally fulfill these to be considered a strong match.  
{wanted_features}

**Recommended Product:**
- Product Information: 
{products_info}

## Evaluation Dimensions

### 1. Query Relevance
Does the product match the user's search intent?

**Scoring Criteria:**
- 1: Product is relevant and matches the user's search intent
- 0: Product is irrelevant or does not match the user's search intent

### 2. Preference Match Count
How many of the user's wanted features does the recommended product have?
- Count and return only the number of matched features.

## Output Format

```json
{{
  "query_relevance": <0 or 1>,
  "preference_match_count": <matched_count>
}}
```
"""
add_on_deals_stage_1_evaluation_prompt = """
You are an expert evaluator for large language model outputs. Your task is to assess a ReAct-style assistant's ability to retrieve user memories and extract user preferences for product recommendations accurately.

## Evaluation Task

Please evaluate the model's output based on the following information:

**User Query:**
{question}

**Model Output (answer):**
{answer}

**Ground Truth (Reference Product Bundle):**
This defines the ideal set of products and their required features.
{reference_product_bundle}

## Evaluation Dimensions

### 1. Query Relevance
- Is the model's answer relevant to the user query?
- Does the answer directly address the recommendation request?

**Scoring Criteria:**
- 1: Answer is relevant and directly addresses the user query
- 0: Answer is irrelevant or does not address the user query

### 2. Product Count
- How many products mentioned in the model's answer are relevant to the reference product bundle?
- Count and return only the number of products that align with the reference bundle.

### 3. Preference Match Count
- How many of the user's wanted features/preferences are mentioned or extracted in the model's answer?
- Count and return only the number of matched features across all products.

## Output Format

Please provide only the evaluation scores in the following JSON format:

```json
{{
  "query_relevance": <0 or 1>,
  "product_count": <matched_products_count>,
  "preference_match_count": <matched_features_count>
}}
```
"""

add_on_deals_stage_2_evaluation_prompt = """You are a professional query-product matching evaluator.

## Task
Given a user query, wanted features, and recommended products, evaluate how well the recommended products match the user's requirements.

## Matching Rules
- Use only the provided attributes; do not infer from product name.
- Treat wanted features as key:value pairs. For each feature, the recommended product must contain the equivalent key and value (case-insensitive).
- Ignore attributes order and any extra attributes in the recommended products.
- If any wanted feature is missing or you are uncertain, do not count it as a match.

## User Query
{user_query}

## User Wanted Features
{wanted_features}

## Recommended Products
{recommended_products}

## Evaluation Dimensions

### 1. Query Relevance Count
How many recommended products are relevant to and match the user query?
- Count and return only the number of products that match the user query intent.

### 2. Feature Match Count
How many of the user's wanted features are present in the recommended products?
- Count and return only the total number of matched features across all recommended products.

## Output Format (JSON only)

```json
{{
  "query_relevance_count": <matched_products_count>,
  "feature_match_count": <matched_features_count>
}}
```
"""

single_product_evaluation_prompt = """You are a professional query-product matching evaluator.

**Task:**
Given a user query, wanted features, and a recommended product, answer "yes" ONLY if BOTH conditions are met:
1. The recommended product is relevant to the user query
2. The recommended product contains ALL wanted features

**Matching Rules:**
- Compare features comprehensively: consider name, attributes, and options of the recommended product
- Match features semantically (exact wording not required, meaning must align)
- Ignore attribute order and extra attributes in the recommended product
- Answer "no" if ANY wanted feature is missing or unclear

**Default to "no" when uncertain.**

** User Query:**
{user_query}

** User Wanted Features:**
{wanted_features}

** Recommended Product:**
{recommended_product}

Output "yes" or "no" without any other text.
"""

add_on_deals_evaluation_prompt = """You are a professional query-product matching evaluator.

** Task:**
Given:
- user query: include multiple products
- wanted features: include features that correspond to the each product in the user query
- recommended products
Answer "yes" ONLY if BOTH conditions are met:
1. The recommended products include ALL the products in the user query
2. For each product in the user query, the recommended product contains ALL the corresponding wanted features

** Matching rules:**
- Compare features comprehensively: consider name, attributes, and options of the recommended product
- Match features semantically (exact wording not required, meaning must align)
- Ignore the product order in the user query and the recommended products
- Ignore the attribute order and extra attributes in the recommended product
- Answer "no" if ANY wanted feature is missing or unclear

**Default to "no" when uncertain.**

** User Query:**
{user_query}

** User Wanted Features:**
{wanted_features}

** Recommended Products:**
{recommended_products}

Output "yes" or "no" without any other text.
"""
