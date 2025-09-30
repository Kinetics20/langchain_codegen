import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

def clean_code(text: str) -> str:
    """Remove markdown code fences (```python ... ```)."""
    # Regex to extract code inside ``` blocks if present
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def generate_code(task: str) -> str:
    """Generate Python code from task description."""
    template = """
    You are Guido van Rossum, the creator of Python.
    Write clean, enterprise-level Python code that solves the following task.
    Always follow PEP8, typing annotations, and the Zen of Python.
    
    Requirements:
    - Include type hints for all function signatures.
    - Add a clear docstring for each function.
    - At the end of the file, include a function call inside:
        if __name__ == "__main__":
            print(...)
      so that running the script will produce output.

    Return ONLY the Python code in a markdown block.

    Task: {task}
    """
    prompt = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"task": task})

    return clean_code(raw_output)


if __name__ == "__main__":
    task = input("Enter task description: ")
    code = generate_code(task)

    print("\n=== Generated Code ===\n")
    print(code)

    filename = "generated_code.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"\nCode saved to file: {filename}")
