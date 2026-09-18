import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ANTHROPIC_MODEL_NAME, GEMINI_MODEL_NAME, resolve_provider


def extract_with_tool(
    system_blocks: list[dict[str, Any]], user_text: str, tool_schema: dict[str, Any]
) -> dict[str, Any]:
    provider = resolve_provider()
    if provider == "anthropic":
        return _extract_with_anthropic(system_blocks, user_text, tool_schema)
    if provider == "gemini":
        return _extract_with_gemini(system_blocks, user_text, tool_schema)
    raise RuntimeError("No LLM provider is configured; use the rule-based fallback instead.")


def _extract_with_anthropic(
    system_blocks: list[dict[str, Any]], user_text: str, tool_schema: dict[str, Any]
) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL_NAME,
        max_tokens=1024,
        system=system_blocks,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": tool_schema["name"]},
        messages=[{"role": "user", "content": user_text}],
    )
    usage = response.usage
    tool_use = next(block for block in response.content if block.type == "tool_use")
    return {
        "extracted": tool_use.input,
        "llm_provider": "anthropic",
        "model_name": ANTHROPIC_MODEL_NAME,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
    }


def _extract_with_gemini(
    system_blocks: list[dict[str, Any]], user_text: str, tool_schema: dict[str, Any]
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client()
    system_text = "\n\n".join(block.get("text", "") for block in system_blocks)
    function_declaration = types.FunctionDeclaration(
        name=tool_schema["name"],
        description=tool_schema.get("description", ""),
        parameters_json_schema=tool_schema["input_schema"],
    )
    tool = types.Tool(function_declarations=[function_declaration])
    config = types.GenerateContentConfig(
        system_instruction=system_text,
        tools=[tool],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME, contents=user_text, config=config
    )
    call = response.candidates[0].content.parts[0].function_call
    return {
        "extracted": dict(call.args),
        "llm_provider": "gemini",
        "model_name": GEMINI_MODEL_NAME,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def judge_llm() -> Any:
    provider = resolve_provider()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(ChatAnthropic(model=ANTHROPIC_MODEL_NAME))
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME))
    raise RuntimeError("No LLM provider is configured; Ragas LLM-judged metrics are skipped.")
