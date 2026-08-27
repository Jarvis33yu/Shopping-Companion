web_visit_summary_prompt = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
2. **Key Extraction for Evidence**: Identify the **most relevant information** from the content, and output the **full original context** containing that information as completely as possible. Ignore the irrelevant information.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

mem_summary_prompt = """Please process the following conversation and user goal to extract relevant information:

## **Conversation**
{conversation}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the conversation.
2. **Key Extraction for Evidence**: Identify the **most relevant information** from the conversation, and output the **full original context** containing that information as completely as possible. Ignore the irrelevant information.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "rational": {
                    "type": "string",
                },
                "evidence": {
                    "type": "string",
                },
                "summary": {
                    "type": "string",
                },
            },
            "required": ["rational", "evidence", "summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
