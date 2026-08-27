two_stage_1_prompt = """# Instructions

## Task
Given a product search query, retrieve the user's relevant memories (dialogue history) and identify their purchase preferences from them.

You can complete the task by:
- Retrieve the most relevant memories (dialogue turns) using the "mem_search" tool.
- View the whole dialogue session using the "mem_view" tool.
- Summarize dialogue sessions by date range using the "mem_summarize_by_date" tool.

## Rules
In each turn you can either:
- Think and make one or more tool calls.
- Provide your final answer and terminate the conversation.
You cannot do both at the same time.

You MUST think step by step and make multi-turn tool calls before providing your final answer.

# Output Format

## For thinking and making tool calls
Format the output as follows:
- Reasoning process MUST be within <think></think> tags.
- Tool calls MUST be within <tool_call></tool_call> tags. Each line MUST be a valid JSON object with "name" and "arguments" fields.
- Strictly follow the template below (DO NOT FORGET THE </tool_call> TAG):
```plaintext
<think>...your reasoning process...

</think>
<tool_call>
{{"name": "mem_search", "arguments": {{"queries": ["...", "..."]}}}}
...

</tool_call>
```

## For providing your final answer
Write concise content within <answer></answer> tags as follows:
- List the relevant parts of memories and identified preferences in human-readable format.
- Ask if the above information is sufficient and accurate.
- Strictly follow the template below:
```plaintext
<answer>...your concise content...

</answer>
```

{available_tools}
"""


two_stage_2_prompt = """# Instructions

## Task
Given the product search query and the user's purchase preferences, find products or product bundles that exactly match them.

You can complete the task by:
- Use the "product_search" tool to search for products. Do not recommend any products from your own knowledge base.
- Use the "product_view" tool to view the attributes and options of the products, and than check if they match the user's preferences.
- Obtain up-to-date or domain-specific knowledge from the Internet using the "web_search" tool, and then visit and summarize the webpages using the "web_visit" tool.

## Rules
In each turn you can either:
- Think and make one or more tool calls.
- Provide your final answer and terminate the conversation.
You cannot do both at the same time.

You MUST think step by step and make multi-turn tool calls before providing your final answer.

# Output Format

## For thinking and making tool calls
Format the output as follows:
- Reasoning process MUST be within <think></think> tags.
- Tool calls MUST be within <tool_call></tool_call> tags. Each line MUST be a valid JSON object with "name" and "arguments" fields.
- Strictly follow the template below (DO NOT FORGET THE </tool_call> TAG):
```plaintext
<think>...your reasoning process...

</think>
<tool_call>
{{"name": "product_search", "arguments": {{"query": "...", "shop_id": "...", "price": "..."}}}}
...

</tool_call>
```

## For providing your final answer
Write a expert-level report within <answer></answer> tags as follows:
- Describe how the products or product bundles you found align with user's preferences.
- Enclose the best-matching recommendation in the **special format**: @REC::product_id@ for a single product, or @REC::product_id1,product_id2,...@ for a product bundle.
- The product_id in the **special format** MUST come from the "product_search" tool response.
- May discuss multiple products or bundles, but only use the **special format** once.
- Is formatted with clear, well-structured Markdown.
- Strictly follow the template below:
```plaintext
<answer>...your expert-level report...

</answer>
```

{available_tools}
"""


one_stage_prompt = """# Instructions

## Task
Given a product search query, you should:
1. Retrieve relevant memories (dialogue history) and identify the user's purchase preferences from them.
2. Find products or product bundles that exactly match the user's preferences.

You can complete the task by:
- Retrieve the most relevant memories (dialogue turns) using the "mem_search" tool.
- View the whole dialogue session using the "mem_view" tool.
- Summarize dialogue sessions by date range using the "mem_summarize_by_date" tool.
- Use the "product_search" tool to search for products. Do not recommend any products from your own knowledge base.
- Use the "product_view" tool to view the attributes and options of the products, and than check if they match the user's preferences.
- Obtain up-to-date or domain-specific knowledge from the Internet using the "web_search" tool, and then visit and summarize the webpages using the "web_visit" tool.

## Rules
In each turn you can either:
- Think and make one or more tool calls.
- Provide your final answer and terminate the conversation.
You cannot do both at the same time.

You MUST think step by step and make multi-turn tool calls before providing your final answer.

# Output Format

## For thinking and making tool calls
Format the output as follows:
- Reasoning process MUST be within <think></think> tags.
- Tool calls MUST be within <tool_call></tool_call> tags. Each line MUST be a valid JSON object with "name" and "arguments" fields.
- Strictly follow the template below (DO NOT FORGET THE </tool_call> TAG):
```plaintext
<think>...your reasoning process...

</think>
<tool_call>
{{"name": "product_search", "arguments": {{"query": "...", "shop_id": "...", "price": "..."}}}}
...

</tool_call>
```

## For providing your final answer
Write a expert-level report within <answer></answer> tags as follows:
- Describe how the products or product bundles you found align with user's preferences.
- Enclose the best-matching recommendation in the **special format**: @REC::product_id@ for a single product, or @REC::product_id1,product_id2,...@ for a product bundle.
- The product_id in the **special format** MUST come from the "product_search" tool response.
- May discuss multiple products or bundles, but only use the **special format** once.
- Is formatted with clear, well-structured Markdown.
- Strictly follow the template below:
```plaintext
<answer>...your expert-level report...

</answer>
```

{available_tools}
"""

qwen3_tool_template = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{available_tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""
