from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os

load_dotenv()

# --- LangSmith Tracing Configuration ---
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "true"))
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGSMITH_ENDPOINT", "https://apac.api.smith.langchain.com"))
os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "first"))


def main():
    print("Hello world")

    information = """
Narendra Damodardas Modi was born on 17 September 1950 to a Gujarati family of Other Backward Class (OBC) background and Hindu faith in Vadnagar, Mehsana district, Bombay State (present-day Gujarat). He was the third of six children born to Damodardas Mulchand Modi (c. 1915-1989) and Hiraben Modi (1923-2022). According to Modi and his neighbours, he worked infrequently in his father's tea stall in the Vadnagar railway station.

Modi completed his higher secondary education in Vadnagar in 1967; his teachers described him as an average student and a keen, gifted debater with an interest in theatre. He preferred playing larger-than-life characters in theatrical productions, which has influenced his political image.
"""

    summary_template = """
    Following is the information {information} about a person for whom you should create a:
    1. short summary of his bio
    2. Interesting facts from the information provided
"""

    final_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperateure=1, model="llama3.1:latest")

    chain = final_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
