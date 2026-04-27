
from typing import List

import instructor
from jinja2 import Template
from langchain_core.messages import convert_to_openai_messages
from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel, Field

from api.agents.tools import get_formatted_context
from api.agents.utils.utils import format_ai_message
from api.agents.utils.prompt_management import prompt_template_config
from api.agents.local_models import CHAT_MODEL, client as openai_compatible_client  , EMBED_MODEL


# QNA Agent Response Model
class ToolCall(BaseModel):
    name: str
    arguments: dict


class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(
        description="Short description of the item used to answer the question"
    )


class AgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: List[RAGUsedContext] = Field(
        default_factory=list,
        description="List of items used to answer the question.",
    )
    final_answer: bool = False
    tool_calls: List[ToolCall] = Field(default_factory=list)


# Intent Router Response Model
class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str


# ONA Agent Node
@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "ollama", "ls_model_name": CHAT_MODEL},
)
def agent_node(state) -> dict:

    template = prompt_template_config("api/agents/prompts/qa_agent.yaml", "agent")

    prompt = template.render(
        available_tools=state.available_tools,
    )

    messages = state.messages

    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(openai_compatible_client)

    response, raw_response = client.chat.completions.create_with_completion(
        model=CHAT_MODEL,
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references,
    }


@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "ollama", "ls_model_name": CHAT_MODEL},
)
def intent_router_node(state) -> dict:
  
    template = prompt_template_config("api/agents/prompts/intent_router_agent.yaml", "intent_router_agent")
    prompt = template.render()

    messages = state.messages

    conversation = []
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(openai_compatible_client)

    response, raw_response = client.chat.completions.create_with_completion(
        model=CHAT_MODEL,
        response_model=IntentRouterResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
    }













# from langsmith import traceable

# from langchain_core.messages import convert_to_openai_messages
# from openai import OpenAI


# from jinja2 import Template
# import instructor
# from api.agens.tools import get_formatted_context
# from api.agens.utils.prompt_management import prompt_template_config
# from api.agens.utils.utils import format_ai_message


# client = OpenAI(
#     base_url="http://ollama:11434/v1",
#     api_key="ollama",
# )


# def agent_node(state : State):
#     prompt_template = """ You are a helpful assistant that can answer questions about the products in stock.

#     You will be given a conversation history and a question and list of tools you can use to answer the latest query.

#     <Available tools>
#     {{available_tools | tojson}}
#     </Available tools>

#     When making tool calls, use the following format:
#     {
#         "name" : "tool_name",
#         "args" : {
#             "parameter1" : "value1",
#             "parameter2" : "value2",
#         }
#     }

#     CRITICAL: All parameters must go inside the "args" object, not at the top level of the tool call. Do not add any parameters outside of it.

#     Examples:
#     - Get formatted item context:
#     {
#         "name" : "get_formatted_item_context",
#         "args" : {
#             "query" : "Cool kids toys",
#             "top_k" : 5
#         }
#     }

#     CRITICAL RULES:
#     - if tool_calls has values, final_answer must be false
#     {You cannot call tools and exit the graph in the same response}
#     - if final_answer is true, tool_calls must be []
#     {You must wait for tool results before exiting the graph }
#     - If you need tool results before answering, set:
#     tool_calls = [...], final_answer = false
#     - After receiving tool results, set:
#     tool_calls = [], final_answer = true, and use the tool results to answer the question
#     - Use names specific provided in the available tools list when making tool calls. Do not make up tool names or use names not in the available tools list.

#     Instructions:
#     - You need to answer the question based on the outputs from the tools using the available tools only.
#     - Do not suggest same tool call more than once.
#     - If the question can be decomposed into multiple sub-questions that can be answered by the tools, suggest all of them.
#     - If multiple tool calls can be used once to answer the question , suggest all of them.
#     - Do not explain your next step in answer, instead use tools to answer the question.
#     - Never use word context and refer to it as the available products.
#     - You should only answer questions about the products in stock. If the question is not about products in stock, you should ask for clarification.
#     - As an output you need return the following:

#     * answer: The answer to the question based on your knowledge and the tool results.
#     * references: The list of the indexes from the chunks returned from all tool calls that were used to answer the question. If more than one chunk was used to compile the answer from a single tool call, be sure to return all of them.
#     * Each reference should have an id and short descripion of the item based on retrieved context.
#     * final_answer: True if you have all the information needed to provide a complete answer, False otherwise.

#     - The answer to the question should contain detailed information about the product and should be returned with detailed specification in bullet points.
#     - The short description should have the name of the item.
#     - If the user's request requires using a tool, set tool_calls with the appropriate names and arguments. 

#     """

#     template = Template(prompt_template)
#     prompt = template.render(available_tools = state.available_tools)

#     messages = state.messages
#     conversation = []
#     for message in messages:
#         conversation.append(convert_to_openai_messages(message))
#         client = instructor.from_provider(
#         "ollama/gemma3:4b",
#             base_url="http://ollama:11434/v1",
#             api_key="ollama",
#         mode = instructor.Mode.JSON,
#     )
    
#         response = client.create(
#          messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful shopping assistant that can answer questions about products based on the available products."
#             },
#             {"role": "user", "content": prompt}
#         ],
#         response_model = RAGGenerationResponse,
#         timeout = 10.0,
#         max_retries= 2,
#     )
#         ai_message = format_ai_message(response)
#         return {
#             "messages" : [ai_message],
#             "tool_calls": response.tool_calls,
#             "iteration": state.iteraation + 1,
#             "answer" : response.answer,
#             "final_answer" : response.final_answer,
#             "references" : response.references
#         }

