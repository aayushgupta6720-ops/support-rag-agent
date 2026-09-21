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
