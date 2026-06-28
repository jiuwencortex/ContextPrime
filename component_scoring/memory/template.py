MEMORY_PROMPT_TEMPLATE = """\
You are given a section from a project memory file.
Generate {n} realistic user queries that this section would help answer.

Each example must have:
  - "query": a natural user message (1-3 sentences) that would benefit from knowing
    the information in this section
  - "answer": what a good response looks like when this information is available
    (1-2 sentences, concrete)

Source file: {source_file}
Section heading: {name}
First sentence: {description}

Section content:
{body}

Requirements:
- Queries must be realistic user questions or task requests, not descriptions of the section.
- Cover the different facts or sub-topics within this section.
- Do not generate queries for information that is not in this section.

Return a JSON array of exactly {n} objects:
[{{"query": "...", "answer": "..."}}, ...]
No other text."""