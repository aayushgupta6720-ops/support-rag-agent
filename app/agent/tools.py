from google.genai import types

SEARCH_DOCS_TOOL_NAME = "search_support_docs"

SEARCH_DOCS_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=SEARCH_DOCS_TOOL_NAME,
            description=(
                "Search internal support documentation for information "
                "relevant to the user's question."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        )
    ]
)

# The tool loop's other two tools (AGENT_MODE=loop; see app/agent/loop.py).
READ_ARTICLE_TOOL_NAME = "read_article"
GIVE_ANSWER_TOOL_NAME = "give_answer"

AGENT_TOOLS = types.Tool(
    function_declarations=[
        *SEARCH_DOCS_TOOL.function_declarations,
        types.FunctionDeclaration(
            name=READ_ARTICLE_TOOL_NAME,
            description=(
                "Read a whole help article by its id, as given with a search "
                "result. For when a search found the right article but not the "
                "passage with the detail you need."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {"article_id": {"type": "string", "description": "The article's id."}},
                "required": ["article_id"],
            },
        ),
        types.FunctionDeclaration(
            name=GIVE_ANSWER_TOOL_NAME,
            description="Give your reply to the user. This ends your turn; always finish with it.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The reply to the user."},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ids of the articles the reply uses; empty if it uses none.",
                    },
                },
                "required": ["answer", "sources"],
            },
        ),
    ]
)
