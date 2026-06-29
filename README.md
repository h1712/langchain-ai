# langchain-ai

AI Study Guide & LangChain Hands-On Examples

## Study Guides

| # | Topic | File |
|---|---|---|
| 1 | [Large Language Models (LLMs)](01-llm-fundamentals.md) | Transformers, attention, GPT architecture, inference, tokenization |
| 2 | [Retrieval-Augmented Generation (RAG)](02-rag-architecture.md) | RAG pipeline, chunking, retrieval strategies, evaluation |
| 3 | [Embeddings & Vector Databases](03-embeddings-vector-db.md) | Embedding models, similarity search, ANN indexes, Milvus/FAISS/Pinecone |
| 4 | [Prompt Engineering](04-prompt-engineering.md) | Techniques, few-shot, chain-of-thought, system prompts, guardrails |
| 5 | [LangChain & Frameworks](05-langchain-framework.md) | Chains, agents, tools, memory, LCEL |
| 6 | [Fine-Tuning & RLHF](06-finetuning-rlhf.md) | SFT, LoRA, RLHF, DPO, reward models |
| 7 | [Ollama vs OpenAI Observations](ollama-vs-openai-observation.md) | Side-by-side comparison of local vs cloud LLM |

## Hands-On Programs

### 1. `simple_llm_demo.py` — LLM Basics with Azure OpenAI
Demonstrates three LangChain patterns:
- Simple `llm.invoke()` call
- Prompt Template + LCEL chain (`prompt | llm | parser`)
- Batch execution (multiple queries in parallel)

```bash
python simple_llm_demo.py
```

### 2. `langchain-basics.py` — Azure OpenAI with PromptTemplate
Uses `PromptTemplate` + LCEL chain to summarize a person's bio and extract interesting facts.

```bash
python langchain-basics.py
```

### 3. `langchain-using-ollma.py` — Local LLM with Ollama
Same logic as above but runs **locally** using Ollama (LLaMA 3.1). No API key needed, fully private.

```bash
ollama pull llama3.1
python langchain-using-ollma.py
```

### 4. `langchain-ollama-langsmith.py` — Ollama + LangSmith Tracing
Same as the Ollama script but with **LangSmith observability** enabled. Traces every chain step (prompt input, LLM call, output) to the LangSmith dashboard.

```bash
# Add to .env: LANGSMITH_API_KEY=lsv2_pt_xxx
python langchain-ollama-langsmith.py
# View traces at https://smith.langchain.com
```

### 5. `langchain_web_search_agent.py` — ReAct Agent with Web Search

An **agentic** program that uses the ReAct pattern (Reasoning + Acting). The LLM decides when to call a web search tool to answer the user's question.

**How it works:**
```
User Query → LLM (decides action) → Calls search tool → Gets result → LLM synthesizes answer
```

**Architecture:**
```
┌─────────────┐      ┌─────────────┐      ┌──────────────┐
│  User Query │ ───► │  LLM Agent  │ ───► │ Tavily Search│
│  (CLI args) │      │  (GPT-4)    │      │ (Web API)    │
└─────────────┘      │             │ ◄─── │              │
                     │  Synthesize │      └──────────────┘
                     │  Answer     │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ Final Answer│
                     └─────────────┘
```

**Features:**
- Custom `@tool` search function wrapping Tavily API
- Also includes `TavilySearch()` as a built-in alternative
- Accepts query from command-line arguments
- LLM autonomously decides whether to call the search tool

**Usage:**
```bash
# Requires TAVILY_API_KEY in .env (get free key at https://tavily.com)
python langchain_web_search_agent.py what is the weather in Tokyo
python langchain_web_search_agent.py latest AI jobs in India
python langchain_web_search_agent.py yocto project latest release
```

**Key Concept — ReAct Agent Loop:**
1. LLM receives the question
2. LLM **reasons**: "I need to search the web for this"
3. LLM **acts**: calls the `search` tool with a query
4. LLM **observes**: reads the search results
5. LLM **answers**: synthesizes a final response from the observations

This is the same pattern used in TejaBot's agentic routing (Jira, Confluence, knowledge search).

### 6. `Increment_calculating_agent.py` — Multi-Tool Agent with Custom Logic

A **ReAct agent** that calculates employee salary increments by chaining multiple tool calls. Demonstrates how an LLM autonomously decides which tools to call and in what order.

**Tools:**
- `get_current_salary(employee)` — looks up salary from a dictionary
- `calculate_incremented_salary(current_salary, rating)` — applies rating-based bonus percentage

**How it works:**
```
Question: "Calculate Ravi's incremented salary (outstanding rating)"
    │
    ▼ Iteration 1
LLM calls: get_current_salary("Ravi") → returns 1000
    │
    ▼ Iteration 2
LLM calls: calculate_incremented_salary(1000, "outstanding") → returns 1120
    │
    ▼ Iteration 3
LLM synthesizes: "Ravi's incremented salary is 1120 (12% bonus for outstanding)"
```

**Key learnings:**
- Ollama `llama3.1` does NOT support structured tool calling properly — outputs JSON text instead of function calls. Commented in code with explanation.
- Azure OpenAI GPT-4 handles `bind_tools()` correctly with proper `tool_calls` format.
- Agent loop processes one tool per iteration for clarity and control.
- `@traceable` decorator enables LangSmith tracing for the full agent loop.

**Usage:**
```bash
python Increment_calculating_agent.py
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-openai langchain-core langchain-ollama langchain-tavily python-dotenv tavily-python
```

### Required `.env` file
```
OPENAI_API_KEY=your-azure-openai-key
TAVILY_API_KEY=tvly-xxxx               # For web search agent
LANGSMITH_API_KEY=lsv2_pt_xxxx         # Optional: for tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com
LANGSMITH_PROJECT=first
```