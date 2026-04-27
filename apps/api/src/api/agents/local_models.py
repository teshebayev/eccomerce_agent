from openai import OpenAI

OLLAMA_BASE_URL = "http://ollama:11434/v1"
OLLAMA_API_KEY = "ollama"

CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)