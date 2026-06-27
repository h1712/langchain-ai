"""
Simple LangChain program to invoke Azure OpenAI LLM.
Demonstrates: LLM call, prompt template, chain, and output parsing.
"""

from dotenv import load_dotenv
load_dotenv()  # Loads .env file automatically

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- Configuration (Azure OpenAI - same as TejaBot) ---
os.environ.setdefault("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://xchat.openai.azure.com")

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment="xchat4",
    api_version="2023-09-15-preview",
    temperature=0,
)

# --- Example 1: Simple LLM call ---
print("=" * 50)
print("Example 1: Simple LLM Call")
print("=" * 50)

response = llm.invoke("What is Retrieval-Augmented Generation in one sentence?")
print(f"Response: {response.content}\n")

# --- Example 2: Prompt Template + Chain (LCEL) ---
print("=" * 50)
print("Example 2: Prompt Template + LCEL Chain")
print("=" * 50)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert. Explain concepts clearly and concisely."),
    ("human", "Explain {topic} in 3 bullet points.")
])

# LCEL chain: prompt → llm → parse output as string
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"role": "AI/ML", "topic": "vector embeddings"})
print(f"Response:\n{result}\n")

# --- Example 3: Batch multiple queries ---
print("=" * 50)
print("Example 3: Batch Execution")
print("=" * 50)

topics = [
    {"role": "AI", "topic": "attention mechanism"},
    {"role": "DevOps", "topic": "Kubernetes pods"},
]

results = chain.batch(topics)
for topic, result in zip(topics, results):
    print(f"\n--- {topic['topic']} ---")
    print(result)

print("\n" + "=" * 50)
print("Done!")
