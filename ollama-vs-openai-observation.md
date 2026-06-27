# Ollama (Local) vs Azure OpenAI — Observation

## Setup
- **Azure OpenAI**: GPT-4 (`xchat4` deployment), temperature=0
- **Ollama (Local)**: LLaMA 3.1 (8B), temperature=1

Same prompt and input text used for both.

---

## Observations

| Aspect | Ollama (LLaMA 3.1 8B) | Azure OpenAI (GPT-4) |
|---|---|---|
| **Response Length** | Shorter, more concise (~150 words) | Longer, more detailed (~250 words) |
| **Structure** | Simple bold headers + bullets | Markdown with `###`, `---` separators, nested bold |
| **Accuracy** | Correct but surface-level | Correct with additional inferences (e.g., mother's age at death) |
| **Consistency** | Varies between runs (temperature=1) | Consistent output (temperature=0) |
| **Interesting Facts** | 3 facts, brief | 5 facts, more elaborated |
| **Tone** | Direct, to-the-point | Polished, structured like a report |
| **Latency** | ~2-4s (local, no network) | ~3-5s (API call over network) |
| **Cost** | Free (runs locally) | Pay per token (~$0.03/1K input, $0.06/1K output for GPT-4) |

---

## Key Takeaways

1. **GPT-4 is more detailed** — produces richer, better-structured output with additional inferences from the text
2. **Ollama is free and private** — no data leaves your machine, good for prototyping and sensitive data
3. **Temperature matters** — Ollama at temp=1 gives varied outputs each run; GPT-4 at temp=0 is deterministic
4. **For production RAG/support bots**: GPT-4 (or GPT-4o) is preferred for quality
5. **For learning/experimentation**: Ollama is perfect — instant feedback, no API costs

---

## Code Difference

```python
# Azure OpenAI (langchain-basics.py)
llm = AzureChatOpenAI(
    azure_deployment="xchat4",
    api_version="2023-09-15-preview",
    temperature=0,
)

# Ollama Local (langchain-using-ollma.py)
llm = ChatOllama(temperature=1, model="llama3.1:latest")
```

**Everything else stays the same** — prompt template, chain, invoke call. This is the power of LangChain's abstraction: swap the LLM provider with one line change.
