SKILL_PROMPT_TEMPLATE = """\
You are given a skill's name, description, and full usage instructions.
Generate {n} realistic examples of this skill being used.

Each example must have:
  - "query": a natural user message (1-3 sentences) that this skill handles
  - "answer": a short description of what this skill would produce (1-2 sentences,
    concrete, describes the result not the process)

Requirements for queries:
- Cover different sub-capabilities from the instructions.
- Vary phrasing and context. No two queries should be structurally identical.
- Stay within scope — do not generate requests the skill cannot handle.
- Use realistic language: how a real user would phrase the request.

Requirements for answers:
- Describe the output ("Meeting recap email sent to sales team with PDF attached").
- Be concrete, include domain-relevant details.
- The answer must be achievable by this skill given the query.

Skill name: {name}
Description: {description}

Instructions:
{body}

Return a JSON array of exactly {n} objects:
[{{"query": "...", "answer": "..."}}, ...]
No other text, no markdown fences."""