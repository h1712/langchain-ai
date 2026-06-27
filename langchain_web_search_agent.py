from dotenv import load_dotenv
from langchain.agents import create_agent
#from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from tavily import TavilyClient

#alternative simple way
from langchain_tavily import TavilySearch

import os
import sys

load_dotenv()

tavily = TavilyClient()

os.environ.setdefault("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://xchat.openai.azure.com")


@tool
def search(query: str) -> str:
    """Tool that searches over internet.

    Arguments:
        query: The question from the user which should be searched over internet

    Returns:
        The search result
    """
    print(f"searching for {query}")
    #return "Weather in tokyo is sunny"
    return tavily.search(query=query)

llm = AzureChatOpenAI(
    azure_deployment="xchat4",
    api_version="2023-09-15-preview",
    temperature=0,
)

#tools = [search]
#alternative instead of search function using inbuily TavilySearch()
tools = [search,TavilySearch()]

agent = create_agent(model=llm, tools=tools)


def main():
    print("hello world")
    user_query=" ".join(sys.argv[1:])
    result = agent.invoke({"messages": [HumanMessage(content=user_query)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
