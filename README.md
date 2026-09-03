<h1 align="center">Vortelio Python SDK</h1>

<p align="center">
  <strong>Run AI on your own computer, from Python, in one line per task.</strong><br/>
  Chat with a language model, generate an image, read a document, speak a sentence — without an API key, without a bill, without sending anything to anyone.
</p>

<p align="center">
  <a href="https://pypi.org/project/vortelio/"><img src="https://img.shields.io/pypi/v/vortelio.svg?style=flat-square" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/vortelio/"><img src="https://img.shields.io/pypi/dm/vortelio.svg?style=flat-square&color=blue" alt="Downloads"/></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/dependencies-0-lightgrey?style=flat-square" alt="Zero dependencies"/>
  <img src="https://img.shields.io/badge/API-OpenAI%20%2B%20Ollama%20compatible-brightgreen?style=flat-square" alt="OpenAI and Ollama compatible"/>
</p>

---

## What it is

[Vortelio](https://github.com/metiu1/Vortelio) is a program that runs AI models on your
own machine — language models, image generation, speech, video, 3D. This package is how
your Python code talks to it.

Normally, doing all of that means a different library for each thing: one for the LLM,
another for Stable Diffusion, another for Whisper, each with its own install, its own
GPU headaches and its own way of being called. Here it is one object with a method per
task, and the models are already running.

```bash
pip install vortelio
```

```python
from vortelio import Vortelio

ai = Vortelio()                                     # http://localhost:11500

ai.pull("llm/mistral:7b")                           # download a model
print(ai.chat("llm/mistral:7b", "What is a lithophane?"))

ai.image("image/sdxl:latest", "a red fox in the snow", "fox.png")
wav = ai.generate_audio("audio/kokoro:latest", "Hello there")
```

## Who it is for

- **People building with the OpenAI or Ollama API** who want to move off the cloud. The
  SDK speaks both wire formats, so existing code keeps working — change the base URL and
  your prompts stop leaving the machine.
- **Anyone who cannot send data out**: client work, medical or legal documents, anything
  under an NDA.
- **Anyone tired of the bill**, or of a rate limit in the middle of a batch job.

## Why this instead of calling the API by hand

- **Zero dependencies.** The sync client imports nothing outside the standard library.
  It will not drag half of PyPI into your project or fight your existing versions.
- **One surface for everything** — chat and streaming, embeddings, RAG, images, video,
  text-to-speech, 3D, agents, MCP and sandboxed code execution, all on the same object.
- **Drop-in compatible** with OpenAI and Ollama clients, so migration is a URL change,
  not a rewrite.
- **It can start the server for you** (`ensure_server()`), so a script works on a fresh
  machine without a manual step.

Async support:

```bash
pip install "vortelio[async]"   # adds aiohttp
```

---

## Prerequisites

Start the Vortelio server first:

```bash
vortelio serve          # default port 11500
```

Or let the SDK auto-start it:

```python
from vortelio import ensure_server
ensure_server()          # finds and starts vortelio if installed
```

---

## Quick Start

```python
from vortelio import Vortelio

ai = Vortelio()          # connects to http://localhost:11500

# Download a model
ai.pull("llm/mistral:7b")

# Chat — streams tokens to stdout, returns full reply
reply = ai.chat("llm/mistral:7b", "What is quantum computing?")

# Generator streaming
for token in ai.chat_stream("llm/mistral:7b", "Tell me a story"):
    print(token, end="", flush=True)
print()
```

---

## Chat & Conversations

```python
# Simple chat
reply = ai.chat("llm/mistral:7b", "Hello!")

# With messages list (Ollama/OpenAI format)
reply = ai.chat("llm/mistral:7b", [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is 2 + 2?"},
])

# Stateful multi-turn conversation
conv = ai.conversation("llm/mistral:7b", system="You are a pirate.")
conv.say("What is your name?")
reply = conv.say("Where do you sail?")

# Streaming from a conversation
for tok in conv.stream("Tell me about treasure"):
    print(tok, end="", flush=True)
```

---

## Generate (Ollama-style)

```python
# Non-streaming
result = ai.generate("llm/mistral:7b", "The capital of France is")
print(result["response"])

# Streaming generator
for tok in ai.generate_stream("llm/mistral:7b", "Count to 10"):
    print(tok, end="", flush=True)

# With options
result = ai.generate(
    "llm/mistral:7b",
    "Explain photosynthesis",
    system="You are a biology teacher.",
    options={"temperature": 0.7, "num_ctx": 4096},
    think=True,   # chain-of-thought with <think> models
)
print(result.get("thinking", ""))
print(result["response"])
```

---

## Embeddings

```python
# Batch embeddings
vecs = ai.embed("llm/nomic-embed-text:latest", ["Hello", "World"])
# → [[0.1, 0.2, ...], [0.3, 0.4, ...]]

# Legacy single-prompt
vec = ai.embeddings("llm/nomic-embed-text:latest", "Hello world")
```

---

## RAG (Retrieval-Augmented Generation)

```python
# Ingest documents
ai.rag_ingest(
    "llm/nomic-embed-text:latest",
    [
        {"text": "Paris is the capital of France.", "meta": {"source": "facts"}},
        {"text": "Berlin is the capital of Germany.", "meta": {"source": "facts"}},
    ],
    collection="my-docs",
)

# Query
hits = ai.rag_query("llm/nomic-embed-text:latest", "capital of France", collection="my-docs")
for h in hits["results"]:
    print(f"[{h['score']:.3f}] {h['text']}")
```

---

## Model Management

```python
ai.models()                         # list all downloaded models
ai.pull("llm/llama3:8b")            # download from HuggingFace
ai.show("llm/mistral:7b")           # model details, template, capabilities
ai.delete("llm/old-model:latest")   # remove a model
ai.copy("llm/mistral:7b", "llm/my-mistral:latest")  # duplicate
ai.quantize("llm/mistral:7b", "q4_k_m")             # quantize
ai.create("llm/my-model:latest", from_model="llm/mistral:7b",
          system="You are a helpful assistant.")
ai.ps()                             # currently loaded models
ai.version()                        # server version
```

---

## Media Generation

```python
# Image
ai.image("image/sdxl:latest", "a red panda on the moon", "panda.png")

# Or get bytes directly
png_bytes = ai.generate_image("image/sdxl:latest", "sunset over mountains")

# Audio (TTS / music)
wav_bytes = ai.generate_audio("audio/kokoro:latest", "Hello, this is a test.")

# Video
mp4_bytes = ai.generate_video("video/wan2-1:latest", "a cat playing piano")

# 3D
obj_bytes = ai.generate_3d("3d/triposr:latest", "a wooden chair")
```

---

## Advanced API

```python
# A/B compare models
result = ai.compare(
    ["llm/mistral:7b", "llm/llama3:8b"],
    "Explain gravity in one sentence.",
)
for r in result["results"]:
    print(f"{r['model']}: {r['response']}")

# Structured JSON output
result = ai.structured(
    "llm/mistral:7b",
    "List 3 programming languages",
    schema={"type": "array", "items": {"type": "string"}},
)
print(result["parsed"])

# Long-text summarization (map-reduce)
summary = ai.summarize("llm/mistral:7b", very_long_text, style="bullets")
print(summary["summary"])

# Chain-of-thought
result = ai.think("llm/qwq:32b", "Is 97 a prime number?")
print("Reasoning:", result["thinking"])
print("Answer:", result["answer"])

# Smart model router
best = ai.route("code", prompt="Write a sorting algorithm")
print("Best model:", best["model"])
```

---

## OpenAI-Compatible API

```python
# Drop-in OpenAI replacement
response = ai.openai_chat(
    "mistral:7b",
    [{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response["choices"][0]["message"]["content"])

# Streaming
for tok in ai.openai_chat_stream("mistral:7b", [{"role":"user","content":"Hi"}]):
    print(tok, end="", flush=True)

# Embeddings (OpenAI format)
result = ai.openai_embeddings("nomic-embed-text:latest", "Hello world")
```

---

## Async Client

```python
import asyncio
from vortelio import AsyncVortelio

async def main():
    ai = AsyncVortelio()

    # All methods are async
    reply = await ai.chat("llm/mistral:7b", "Hello!")

    # Async streaming
    async for tok in ai.chat_stream("llm/mistral:7b", "Tell me a joke"):
        print(tok, end="", flush=True)

    # Async conversation
    conv = ai.conversation("llm/mistral:7b", system="You are helpful.")
    reply = await conv.say("My name is Alice.")

asyncio.run(main())
```

---

## Agents

```python
# List available agents (Open WebUI, OpenClaw, CrewAI, AnythingLLM, ...)
catalog = ai.agents_catalog()

# Install and start an agent
ai.agents_install("open-webui")
ai.agents_start("open-webui")

# Stop an agent
ai.agents_stop("open-webui")
```

---

## Webhooks & Audit

```python
# Register a webhook
ai.hooks_create("https://my-server.com/webhook", event="generate")

# List webhooks
ai.hooks_list()

# Audit log
entries = ai.audit(limit=50)
```

---

## GGUF Inspect & Ollama Import

```python
# Inspect a local GGUF file
info = ai.gguf_inspect("/path/to/model.gguf")

# Import models from a local Ollama installation
ai.import_ollama()   # imports all
ai.import_ollama(["mistral:7b", "llama3:8b"])  # selective
```

---

## HuggingFace Search & Uploads

```python
# Search HuggingFace for models
hits = ai.hf_search("mistral", sort="downloads", gguf=True)
for m in hits:
    print(m["id"], m.get("downloads"))

# Upload a local GGUF to the server
ai.upload("/path/to/model.gguf")

# Detailed local-model metadata
ai.model_info("llm/mistral:7b")
```

---

## Chat History

```python
ai.history(n=20)        # recent generation history
ai.history_clear()      # wipe it

ai.chats()              # saved conversations
ai.chat_get("conv-id")  # one conversation
ai.chat_delete("conv-id")
```

---

## Skills, MCP & Agentic Tools

```python
# Skills
ai.skills()
ai.skill_create("translate", "Translate to Italian", "You translate text to Italian.")
ai.skill_delete("translate")

# MCP servers
ai.mcp_servers()
ai.mcp_add({"name": "fs", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]})
ai.mcp_enable("fs", True)
ai.mcp_remove("fs")

# Agentic generate (tools + autonomous loop)
result = ai.generate(
    "llm/mistral:7b",
    "What time is it and what is 17*23?",
    agentic={"builtins": True, "auto": True},
)

# Approve / answer pending agentic prompts (from a streaming run)
ai.agentic_approve("req-id", approved=True)
ai.agentic_answer("req-id", "yes, use Postgres")
```

---

## Code Execution & File Browse

```python
# Run code in a sandbox (python, js, bash, powershell, go, ruby, php, java, c, cpp, ...)
out = ai.run_code("python", "print(2 ** 10)")
print(out["output"])          # "1024"

# Browse the server's filesystem
ai.fs_list("/home/user")       # entries; empty path → drive roots
ai.fs_read("/home/user/notes.txt")
```

---

## Cloud & Proxy Providers

```python
# Bring-your-own-key cloud LLMs
ai.cloud_providers()
ai.cloud_key("openai", "sk-...")
ai.cloud_chat("openai", "gpt-4o", "Hello!")

# Vortelio managed cloud proxy
ai.proxy_models()
ai.proxy_chat("gpt-4o", "Hello!")
ai.proxy_usage()

# Cloud media providers
ai.media_providers()
ai.media_key("replicate", "r8_...")
```

---

## OpenAI-Compatible Media

```python
# Text-to-speech → saves speech.wav and returns bytes
ai.openai_speech("audio/kokoro:latest", "Hello world", "speech.wav")

# Image generation (OpenAI format)
res = ai.openai_image("image/sdxl:latest", "a neon city", response_format="b64_json")

# Speech translation to English
text = ai.translate("clip.wav")
```

---

## Server Config & Updates

```python
ai.config()                          # current server config
ai.set_config({"theme": "dark"})

ai.update_check()                    # is a newer server available?
ai.update_start(restart=True)        # download + apply
ai.shutdown()                        # stop the server

ai.metrics()                         # Prometheus metrics text
ai.openapi()                         # full endpoint schema
```

---

## Custom Port / Remote Server

```python
ai = Vortelio(host="http://192.168.1.100", port=11500)
ai = Vortelio(port=8080)               # local custom port
ai = Vortelio(timeout=600)             # longer timeout for large models
```

---

## Server Version Compatibility

This SDK version **6.1.0** requires Vortelio server **≥ 0.3.49**.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
