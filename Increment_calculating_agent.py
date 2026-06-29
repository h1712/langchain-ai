from dotenv import load_dotenv

load_dotenv()

# init_chat_model allows easy model switching without vendor-specific imports
# but Ollama llama3.1 doesn't support structured tool calling properly —
# it outputs tool calls as plain text JSON instead of using the function-calling format.
# from langchain.chat_models import init_chat_model

from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
import os

os.environ.setdefault("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://xchat.openai.azure.com")

MAX_ITERATIONS = 10
# MODEL = "llama3.1"  # Ollama local model — doesn't support tool_calls properly with bind_tools()


@tool
def get_current_salary(employee: str) -> float:
    """Look up the salary of an employee in the data provided."""
    print(f"Retrieving current salary of employee {employee}")
    salaries = {"ravi": 1000, "raju": 2000, "laxman": 3000}
    return salaries.get(employee.lower(), 0)


@tool
def calculate_incremented_salary(current_salary: float, rating: str) -> float:
    """Based on the rating assigned calculate the incremented salary.
    Available rating classes: outstanding, effective, poor."""
    print(f"Calculating incremented salary: current_salary={current_salary}, rating={rating}")
    bonus_classes = {"outstanding": 12, "effective": 10, "poor": 8}
    bonus = bonus_classes.get(rating.lower(), 0)
    incremented_salary = round(current_salary * (1 + bonus / 100), 2)
    return incremented_salary


@traceable(name="Langchain agent loop")
def run_agent(question: str):
    tools = [get_current_salary, calculate_incremented_salary]
    tools_dict = {t.name: t for t in tools}

    # Ollama llama3.1 doesn't produce proper tool_calls — outputs JSON text instead
    # llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    # Using Azure OpenAI GPT-4 which supports structured tool calling correctly
    llm = AzureChatOpenAI(
        azure_deployment="xchat4",
        api_version="2023-09-15-preview",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("*" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful calculating agent. "
                "You have access to employee salary details and employee rating. "
                "You have rating vs increment bonus value. "
                "STRICT RULES you must follow: "
                "Never guess or assume - use the tools provided. "
                "Get salary from get_current_salary. "
                "Calculate salary using calculate_incremented_salary. "
                "If some details are not provided, prompt the user to provide them."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration --- {iteration}")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        # Process only first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"Tool result = {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Error: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello world")
    result = run_agent("Hi can you calculate Ravi incremented salary who got outstanding rating")
