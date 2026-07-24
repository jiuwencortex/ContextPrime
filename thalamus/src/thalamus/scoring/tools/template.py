TOOL_PROMPT_TEMPLATE = """\
You are given an AI agent tool — its name, description, and parameters.
Generate {n} realistic user requests where this tool would be useful.

Each example must have:
  - "query": a natural user message (1-3 sentences) where using this tool
    would help produce a good response
  - "answer": what a good response looks like when the tool is available
    (1-2 sentences, concrete, describes the result)

Tool name: {name}
Description: {description}

Tool details:
{body}

Requirements:
- Queries must be realistic user requests, not descriptions of the tool itself.
- Cover different use cases and parameter combinations of this tool.
- Do not generate queries that are outside the scope of what this tool does.

Return a JSON array of exactly {n} objects:
[{{"query": "...", "answer": "..."}}, ...]
No other text."""
