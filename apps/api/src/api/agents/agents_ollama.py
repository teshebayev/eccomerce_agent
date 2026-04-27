import json
import os
from typing import Any, Dict, List

from jinja2 import Template
from langchain_core.messages import AIMessage, convert_to_openai_messages
from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel, Field

from api.agents.utils.prompt_management import prompt_template_config


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")

raw_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(
        description="Short description of the item used to answer the question"
    )


class AgentResponse(BaseModel):
    answer: str = Field(default="", description="Answer to the question.")
    references: List[RAGUsedContext] = Field(
        default_factory=list,
        description="List of items used to answer the question.",
    )
    final_answer: bool = False
    tool_calls: List[ToolCall] = Field(default_factory=list)


class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str = ""


AGENT_PROMPT_FALLBACK = """You are a shopping assistant that can answer questions about the products in stock.

You will be given a conversation history and a list of tools you can use to answer the latest query.

<Available tools>
{{ available_tools | tojson }}
</Available tools>

Return ONLY valid JSON.

Preferred full schema:
{
  "answer": "string",
  "references": [
    {
      "id": "string",
      "description": "string"
    }
  ],
  "final_answer": true,
  "tool_calls": [
    {
      "name": "tool_name",
      "arguments": {
        "parameter1": "value1"
      }
    }
  ]
}

If you need a tool before answering:
- set "answer" to ""
- set "references" to []
- set "final_answer" to false
- put tool calls in "tool_calls"

If you already have enough information:
- set "final_answer" to true
- set "tool_calls" to []
- answer using only the available products

Rules:
- Use only tool results to answer product questions
- Never mention the word "context"; refer to it as "available products"
- Only answer questions about products in stock
- Do not explain your next steps
- Do not return markdown
- Do not return code fences
- Tool call parameters must be inside "arguments"
- Use tool names exactly as provided in available tools

Compatibility note:
If you cannot produce the full schema, you may return a single tool call object only in this format:
{
  "name": "tool_name",
  "arguments": {
    "query": "value",
    "top_k": 5
  }
}
"""


INTENT_ROUTER_PROMPT_FALLBACK = """You are part of a shopping assistant that can answer questions about products in stock.

Return ONLY valid JSON with this exact schema:
{
  "question_relevant": true,
  "answer": ""
}

Rules:
- If the user asks about products in stock, return {"question_relevant": true, "answer": ""}
- If the question is not about products in stock, return {"question_relevant": false, "answer": "short explanation"}
- Do not return markdown
- Do not return code fences
"""


def _load_prompt(yaml_path: str, prompt_key: str, fallback: str) -> Template:
    try:
        return prompt_template_config(yaml_path, prompt_key)
    except Exception:
        return Template(fallback)


def _flatten_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    conversation: List[Dict[str, Any]] = []

    for message in messages:
        if isinstance(message, dict):
            conversation.append(message)
            continue

        converted = convert_to_openai_messages(message)
        if isinstance(converted, list):
            conversation.extend(converted)
        else:
            conversation.append(converted)

    cleaned: List[Dict[str, Any]] = []
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name")

        normalized = {"role": role, "content": content if content is not None else ""}
        if tool_call_id:
            normalized["tool_call_id"] = tool_call_id
        if name:
            normalized["name"] = name

        if role:
            cleaned.append(normalized)

    return cleaned

import json


def _escape_control_chars_in_strings(s: str) -> str:
    """
    Escapes raw control characters that appear inside JSON strings.
    This makes model-produced "almost JSON" parseable by json.loads.
    """
    result = []
    in_string = False
    escape = False

    for ch in s:
        if escape:
            result.append(ch)
            escape = False
            continue

        if ch == "\\":
            result.append(ch)
            escape = True
            continue

        if ch == '"':
            result.append(ch)
            in_string = not in_string
            continue

        if in_string:
            if ch == "\n":
                result.append("\\n")
                continue
            elif ch == "\r":
                result.append("\\r")
                continue
            elif ch == "\t":
                result.append("\\t")
                continue
            elif ord(ch) < 32:
                result.append(f"\\u{ord(ch):04x}")
                continue

        result.append(ch)

    return "".join(result)


def _extract_json(text: str) -> dict:
    text = text.strip()

    # remove markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in model output: {text}")

    candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = _escape_control_chars_in_strings(candidate)
        return json.loads(cleaned)


def _normalize_agent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "name" in payload and "arguments" in payload and "tool_calls" not in payload:
        return {
            "answer": "",
            "references": [],
            "final_answer": False,
            "tool_calls": [
                {
                    "name": payload["name"],
                    "arguments": payload.get("arguments", {}),
                }
            ],
        }

    payload.setdefault("answer", "")
    payload.setdefault("references", [])
    payload.setdefault("final_answer", False)
    payload.setdefault("tool_calls", [])
    return payload


def _build_ai_message(response: AgentResponse) -> AIMessage:
    tool_calls = []
    for i, tc in enumerate(response.tool_calls):
        tool_calls.append(
            {
                "name": tc.name,
                "args": tc.arguments,
                "id": f"call_{i}",
                "type": "tool_call",
            }
        )

    return AIMessage(
        content=response.answer or "",
        tool_calls=tool_calls,
    )


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "ollama", "ls_model_name": CHAT_MODEL},
)
def agent_node(state) -> dict:
    template = _load_prompt(
        "api/agents/prompts/qa_agent.yaml",
        "agent",
        AGENT_PROMPT_FALLBACK,
    )
    prompt = template.render(available_tools=state.available_tools)

    conversation = _flatten_messages(state.messages)

    response = raw_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    payload = _normalize_agent_payload(_extract_json(content))
    parsed = AgentResponse.model_validate(payload)
    ai_message = _build_ai_message(parsed)

    return {
        "messages": [ai_message],
        "tool_calls": parsed.tool_calls,
        "iteration": state.iteration + 1,
        "answer": parsed.answer,
        "final_answer": parsed.final_answer,
        "references": parsed.references,
    }


@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "ollama", "ls_model_name": CHAT_MODEL},
)
def intent_router_node(state) -> dict:
    template = _load_prompt(
        "api/agents/prompts/intent_router_agent.yaml",
        "intent_router",
        INTENT_ROUTER_PROMPT_FALLBACK,
    )
    prompt = template.render()

    conversation = _flatten_messages(state.messages)

    response = raw_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )

    content = response.choices[0].message.content
    payload = _extract_json(content)
    payload.setdefault("answer", "")
    parsed = IntentRouterResponse.model_validate(payload)

    return {
        "question_relevant": parsed.question_relevant,
        "answer": parsed.answer,
    }
