note_prompt = """Analyze the following conversation and generate a structured output in JSON format.

Guidelines:

1. Keywords:
   - Extract 2 to 6 specific, distinct terms that represent the core topics or concepts discussed.
   - Prioritize nouns and noun phrases (e.g., "climate policy", "supply chain").
   - Exclude speaker names, generic time references (e.g., "today", "next week"), and overly vague terms (e.g., "thing", "issue", "conversation").
   - Order keywords from most to least central to the conversation.

2. Facts:
   - Extract 2 to 6 explicit, verifiable statements directly stated in the conversation.
   - Each fact must be a complete sentence, objectively phrased, and grounded in the text (no inference or interpretation).
   - Include only unique facts—avoid paraphrasing the same information multiple times.
   - Focus on concrete details involving date, location, event, quantity, person, organization, or action.

If the conversation contains insufficient information to meet these criteria, return empty arrays for both fields.

Output ONLY a valid JSON object with the following structure:
{
  "keywords": ["keyword1", "keyword2", ...],
  "facts": ["Fact statement 1.", "Fact statement 2.", ...]
}

Conversation for analysis:
"""