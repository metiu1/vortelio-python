"""
Vortelio — Python SDK for local AI.

Run LLMs, generate images, audio, video, and 3D models — all on your own machine.
OpenAI API and Ollama API compatible.

Quick start::

    from vortelio import Vortelio

    ai = Vortelio()
    ai.pull("llm/mistral:7b")

    # Chat (streams to stdout, returns full reply)
    reply = ai.chat("llm/mistral:7b", "What is quantum computing?")

    # Generator streaming
    for token in ai.chat_stream("llm/mistral:7b", "Tell me a story"):
        print(token, end="", flush=True)

    # Stateful conversation
    conv = ai.conversation("llm/mistral:7b", system="You are a helpful assistant.")
    conv.say("My name is Alice.")
    reply = conv.say("What is my name?")

    # Embeddings
    vecs = ai.embed("llm/nomic-embed-text:latest", ["Hello", "World"])

    # RAG
    ai.rag_ingest("llm/nomic-embed-text:latest",
                  [{"text": "Paris is the capital of France.", "meta": {"source": "facts"}}])
    hits = ai.rag_query("llm/nomic-embed-text:latest", "capital of France")

    # Image generation
    ai.image("image/sdxl:latest", "a red panda on the moon", "panda.png")

    # Async client
    from vortelio import AsyncVortelio
    import asyncio

    async def main():
        ai = AsyncVortelio()
        reply = await ai.chat("llm/mistral:7b", "Hello!")
        async for tok in ai.chat_stream("llm/mistral:7b", "Tell me a joke"):
            print(tok, end="", flush=True)

    asyncio.run(main())
"""

from .client import Conversation, Vortelio
from .async_client import AsyncConversation, AsyncVortelio
from ._http import VortElioError
from .setup import (
    ensure_server,
    find_vortelio_exe,
    install_vortelio,
    is_server_running,
)

__version__ = "6.1.0"
__author__ = "Vortelio Contributors"
__license__ = "Apache-2.0"

__all__ = [
    "Vortelio",
    "Conversation",
    "AsyncVortelio",
    "AsyncConversation",
    "VortElioError",
    "ensure_server",
    "find_vortelio_exe",
    "install_vortelio",
    "is_server_running",
    "__version__",
]
