# 5. LangChain & LLM Frameworks

## What Is LangChain?

LangChain is a Python/JavaScript framework for building applications powered by LLMs. It provides abstractions for: connecting LLMs to external data, chaining multiple LLM calls, building agents with tool use, and managing conversation memory.

**Core philosophy:** LLMs are more powerful when connected to other systems (databases, APIs, tools) rather than used in isolation.

---

## LangChain Architecture

```
┌────────────────────────────────────────────┐
│              Your Application              │
├────────────────────────────────────────────┤
│  Chains / Agents / LCEL Pipelines          │
├───────┬───────┬────────┬──────┬────────────┤
│ LLMs  │Prompts│Indexes │Memory│  Tools     │
│       │       │(Vector)│      │            │
├───────┴───────┴────────┴──────┴────────────┤
│           Model Providers / APIs            │
│  (OpenAI, Azure, Anthropic, HuggingFace)   │
└────────────────────────────────────────────┘
```

---

## Core Components

### 1. LLMs & Chat Models

```python
# Direct LLM call
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    deployment_name="xchat4",
    temperature=0,
    api_version="2023-09-15-preview"
)
response = llm.invoke("What is RAG?")
```

**Chat Models** (GPT-4, Claude) take a list of messages:
```python
from langchain.schema import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a technical assistant."),
    HumanMessage(content="Explain IVF_FLAT indexing.")
]
response = llm.invoke(messages)
```

### 2. Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert."),
    ("human", "Explain {topic} in simple terms.")
])

# Fill in variables
formatted = prompt.format_messages(role="AI", topic="embeddings")
```

### 3. Output Parsers

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class TicketClassification(BaseModel):
    category: str
    priority: str
    confidence: float

parser = PydanticOutputParser(pydantic_object=TicketClassification)
# Parser generates format instructions for the prompt
# and validates LLM output against the schema
```

### 4. Chains (Legacy)

Sequential processing — output of one step feeds into the next:

```python
# Old-style chain (still works but LCEL is preferred)
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(role="DevOps", topic="Kubernetes")
```

### 5. LCEL (LangChain Expression Language) — Modern Approach

Pipe operator (`|`) for composing chains:

```python
from langchain_core.output_parsers import StrOutputParser

# Simple chain using LCEL
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"role": "AI", "topic": "RAG"})
```

**LCEL advantages:**
- Streaming support built-in
- Batch/async execution
- Easier debugging (each step is inspectable)
- Automatic retries

### 6. Retrievers

```python
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Milvus(
    embedding_function=embeddings,
    collection_name="slackbot_conv_channel123",
    connection_args={"host": "localhost", "port": "19530"}
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

docs = retriever.invoke("How to fix build error?")
```

### 7. Memory

Stores conversation history across multiple turns:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "What is RAG?"},
    {"output": "RAG is Retrieval-Augmented Generation..."}
)

# Memory types:
# ConversationBufferMemory     — stores all messages (unbounded)
# ConversationBufferWindowMemory — last K messages only
# ConversationSummaryMemory    — LLM summarizes history
# ConversationTokenBufferMemory — bounded by token count
```

### 8. Agents & Tools

Agents decide which tools to use based on the query:

```python
from langchain.agents import create_react_agent, Tool

tools = [
    Tool(name="search_kb", func=search_knowledge_base,
         description="Search the team's knowledge base for technical Q&A"),
    Tool(name="jira_lookup", func=lookup_jira_issue,
         description="Look up a Jira issue by key (e.g., RDK-12345)"),
    Tool(name="create_ticket", func=create_jira_ticket,
         description="Create a new Jira ticket"),
]

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
result = agent.invoke({"input": "Find info about build failures and create a ticket"})
```

**Agent loop:**
```
User Input → LLM decides action → Execute tool → Observe result → 
LLM decides next action → ... → Final Answer
```

---

## RAG Chain with LCEL

The standard RAG pattern in LangChain:

```python
from langchain_core.runnables import RunnablePassthrough

# Build RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use it
answer = rag_chain.invoke("How to fix a segmentation fault in RDK?")
```

This is equivalent to:
1. Retrieve relevant docs from vector store
2. Format prompt with context + question
3. Send to LLM
4. Parse output as string

---

## LangChain vs Other Frameworks

| Feature | LangChain | LlamaIndex | Haystack | Direct API |
|---|---|---|---|---|
| RAG focus | General | Primary | Primary | Manual |
| Agent support | Strong | Basic | Basic | Manual |
| Complexity | High | Medium | Medium | Low |
| Abstraction | Heavy | Medium | Medium | None |
| Debugging | Harder | Easier | Easier | Easiest |
| Community | Largest | Growing | Mature | N/A |

### LlamaIndex (Alternative)
- Focused specifically on RAG / data indexing
- Simpler API for document ingestion and querying
- Better for: "I just want to query my documents"

### Direct API (What TejaBot Uses)
- No framework — direct OpenAI/Milvus API calls
- Maximum control, no framework overhead
- Better for: production systems with specific requirements

---

## LangSmith (Observability)

LangChain's monitoring and debugging platform:

```python
# Automatic tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."

# All chain/agent executions are now traced
# View at: smith.langchain.com
```

**What it shows:**
- Full execution trace of each chain step
- Token usage and latency per step
- Input/output at each stage
- Error diagnosis

---

## Key LangChain Patterns

### 1. Conversational RAG
```python
# RAG + Memory = Conversational RAG
chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: memory.load_memory_variables({})
    )
    | contextualize_prompt  # Rewrite query using chat history
    | retriever
    | qa_prompt
    | llm
)
```

### 2. Multi-Query Retriever
```python
from langchain.retrievers import MultiQueryRetriever

# LLM generates 3 variations of the query → retrieves for all → deduplicates
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, llm=llm
)
```

### 3. Self-Query Retriever
```python
from langchain.retrievers import SelfQueryRetriever

# LLM extracts metadata filters from natural language
# "Find bugs from last week" → filter: {type: "bug", date: "> 7 days ago"}
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info)
```

### 4. Ensemble Retriever
```python
from langchain.retrievers import EnsembleRetriever

# Combine dense + sparse retrieval
ensemble = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)
```

---

## Quick-Recall Points

- LangChain = framework for LLM apps (chains, agents, memory, tools)
- LCEL (pipe operator `|`) is the modern way to compose chains
- Agents = LLM decides which tool to call (ReAct pattern)
- Memory types: Buffer (all), Window (last K), Summary (LLM summarizes)
- Retrievers wrap vector stores with a standard `.invoke()` interface
- LlamaIndex is simpler but narrower (RAG-focused)
- Direct API calls (no framework) give maximum control
- LangSmith provides observability (traces, token usage, latency)

---

## ⭐ Interview / Exam Q&A

**Q: What is LangChain and when would you use it?**
> LangChain is a framework for building LLM-powered applications. Use it when you need: RAG with standard patterns, agent-based tool use, conversation memory, or rapid prototyping. Don't use it for: simple one-off LLM calls, or production systems needing maximum performance control (use direct API instead).

**Q: What is LCEL and why was it introduced?**
> LCEL (LangChain Expression Language) uses the pipe operator to compose chains: `prompt | llm | parser`. It replaced the legacy Chain classes because it provides: built-in streaming, async support, batch execution, and better composability. Each step in a LCEL pipeline is a "Runnable" with a standard interface.

**Q: Explain the difference between a Chain and an Agent.**
> A Chain follows a fixed sequence of steps — every input goes through the same pipeline. An Agent uses an LLM to dynamically decide which step to take next based on the input and observations. Chains are deterministic; Agents are dynamic. Agents can loop, retry, and choose different tools per query.

**Q: How does LangChain handle conversation memory?**
> Memory objects store and retrieve conversation history. BufferMemory stores all messages (grows unbounded). WindowMemory stores last K exchanges. SummaryMemory uses an LLM to compress history into a summary. TokenBufferMemory caps by token count. The memory is injected into the prompt template as a variable on each turn.

**Q: Why does your TejaBot NOT use LangChain?**
> TejaBot uses direct API calls to Azure OpenAI and Milvus for maximum control. Specific reasons: (1) Custom dual-vector search with per-field weighting isn't natively supported by LangChain's Milvus wrapper, (2) Custom three-tier reranking formula needed low-level access, (3) Framework overhead and abstraction layers add latency in a real-time Slack bot, (4) Debugging is easier with direct calls in production.

**Q: What is a Multi-Query Retriever?**
> Instead of searching with just the user's original query, the LLM generates 3–5 query variations capturing different perspectives. All variations are searched, and results are deduplicated. This improves recall because the original query might miss relevant docs that a rephrased version would catch. Example: "RDK build failure" → also searches "compilation error in RDK", "RDK make not working".
