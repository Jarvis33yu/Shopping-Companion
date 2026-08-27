single_product_dialogue_prompt = """<instructions>
You are a professional scriptwriter.

**Input:**
- <product_name>: The product being discussed.
- <features>: Features of the product.

**Tasks:**
1. Choose {number_of_features} meaningful and easy-to-understand features from <features> as **wanted features** and treat the rest as **does not matter features**.
2. Craft a realistic user-assistant dialogue.

**Dialogue Rules:**
1. **Role Clarity**:
   - The user proposes a life scenario where the product would be needed.
   - The assistant knows well the scenario and the product.
2. **User Behavior**:
   - Describe the usage scenario and explain why they want the product.
   - Should not voluntarily propose any features, but passively respond to the those that the assistant proposes.
   - Clearly states whether a feature is **wanted** or **does not matter** through personal experiences.
   - Confirm all the wanted features before the conversation ends, but do not buy the product now.
3. **Assistant Strategy**:
   - Do not make the conversation too salesy. The assistant's goal is to help the user analyze the demand and make a purchase decision.
   - Ask about one aspect at a time to avoid overwhelming the user.
   - Proactively ask clarification questions while the user clarifies the needs.
   - Discuss only features that appear in <features>; exclude features mentioned only in the <product_name>.
4. **Conversation Flow**:
   - Natural, coherent, diverse, and contextually consistent.
   - Reorganize and sequence the attributes logically to reflect a realistic discovery process.
   - Only output utterances, no non-verbal actions.
</instructions>

<product_name>
{product_name}
</product_name>

<features>
{features}
</features>

<output_format>
Return as a valid JSON object with "wanted_features", "does_not_matter_features", and "dialogue" keys:
- "wanted_features": a valid JSON array of strings, each string is a feature that the user wants.
- "does_not_matter_features": a valid JSON array of strings, each string is a feature that the user does not matter.
- "dialogue": a valid JSON array of JSON objects, each object has "role" and "content" keys.
Example format:
```json
{{
    "wanted_features": ["f1: v1", "f2: v2", ...],
    "does_not_matter_features": ["f3: v3", "f4: v4", ...],
    "dialogue": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        ...
    ]
}}
```
</output_format>
"""


single_product_dialogue_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "wanted_features": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "does_not_matter_features": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "dialogue": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                            },
                            "content": {
                                "type": "string",
                            },
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["wanted_features", "does_not_matter_features", "dialogue"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


cycle_reasoning_prompt = """Based on the following product information, determine if this is a fast-moving product with a repeat purchase cycle:
- if so, infer the cycle: two weeks, three weeks, one month, or two months.
- if not, return "no repeat purchase cycle".

Product Name: {product_name}
Category: {category}
Price: ${price}

Answer in JSON format with two keys:
- "think": your reasoning about the product and the repeat purchase cycle.
- "cycle": the repeat purchase cycle.

Example response when valid:
{{
   "think": "...",
   "cycle": "choose one from two weeks, three weeks, one month, two months"
}}
"""


repeat_purchase_dialogue_prompt = """Based on the following product, generate multiple dialogues between a user and an AI shopping assistant 
that show the user's repeat purchase behavior over time.

Product Name: {product_name}
Category: {category}
Price: ${price}
Repeat Purchase Cycle: {cycle}

Generate 3-4 separate conversations that occur at different time intervals (following the repeat purchase cycle):
1. First Dialogue: The initial purchase - user discovers and buys the product for the first time
2. Second Dialogue: The first repeat purchase - user repurchases after using up the product (after {cycle})
3. Third Dialogue: The second repeat purchase - user is now a regular customer (after another {cycle})
4. (Optional) Fourth Dialogue: Another repeat purchase - user continues the regular purchase pattern

Each dialogue should:
- Show the progression of the user becoming more familiar with the product
- Demonstrate why the user keeps coming back to repurchase
- Include specific details about usage, consumption, or depletion
- Show increasing familiarity/comfort in the conversation (from exploratory to confident)
- Each purchase is for a SINGLE unit/bottle/package of {product_name}
- END with the user explicitly confirming they have PURCHASED {product_name}
- END with the assistant acknowledging the successful purchase
- No personal details (addresses, phone, email, payment info)
- No fake order numbers or receipts

Output format: Return as a JSON object where each key represents a purchase occasion and contains a list of message objects:
{{
    "purchase_1": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        ...
    ],
    "purchase_2": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        ...
    ],
    "purchase_3": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        ...
    ],
    "purchase_4": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        ...
    ]
}}

Important: Generate 3-4 purchase dialogues based on what makes sense for the product. Each key should contain a list of message objects with "role" and "content" keys.
"""


complement_dialogue_prompt = """Based on the following product, generate a realistic dialogue between a user and an AI shopping assistant. 
The dialogue should revolve around the user's purchase decision, their needs, and why they chose this product.

Product Name: {product_name}
Category: {category}
Price: ${price}

Generate a natural multi-turn conversation (8-10 turns) where:
1. User mentions their need and asks for help
2. Assistant asks clarifying questions
3. User explains their situation and requirements
4. Assistant recommends {product_name} with relevant details
5. User asks follow-up questions about the product
6. Assistant addresses concerns and confirms suitability
7. User confirms they have already purchased {product_name}
8. Assistant acknowledges the purchase positively

The dialogue should include:
- What problem the product solves
- Why {product_name} fits the user's needs
- Key features and price discussion
- User's satisfaction with the purchase

IMPORTANT GUIDELINES:
- Only discuss {product_name} and its features
- No complementary products, bundles, or alternatives
- Assistant is a shopping guide, not a payment processor
- No personal details (addresses, phone, email, payment info)
- No fake order numbers or receipts
- End with user confirming they already purchased the product
- Assistant responds with encouragement

Output format: Return as a JSON array of message objects with "role" and "content" keys.
Example format:
[
   {{"role": "user", "content": "..."}},
   {{"role": "assistant", "content": "..."}},
   ...
]
"""
