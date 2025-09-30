import os
import re
import ast
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from style_knowledge_base import add_documents, retrieve_style

# Load environment variables from .env
load_dotenv()


def clean_code(text: str) -> str:
    """Remove markdown code fences (```python ... ```)."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate_code(code: str) -> bool:
    """Check if the generated code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in generated code: {e}")
        return False


def list_exported_functions(code: str) -> list[str]:
    """Return names of top-level functions defined in the code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    funcs: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
    return funcs


def list_raised_exceptions(code: str) -> list[str]:
    """Return a sorted list of exception type names that the code explicitly raises."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    exceptions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            name = None
            exc = node.exc
            if isinstance(exc, ast.Call):
                # raise TypeError("msg")
                func = exc.func
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
            elif isinstance(exc, ast.Name):
                # raise TypeError
                name = exc.id
            if name:
                exceptions.add(name)
    return sorted(exceptions)


def generate_code(task: str, llm: ChatOpenAI) -> str:
    """Generate Python code from task description with style context."""
    # Ensure style docs are loaded
    add_documents()
    style_guidelines = retrieve_style(task)
    style_text = "\n".join([doc.page_content for doc in style_guidelines])

    template = f"""
    You are Guido van Rossum, the creator of Python.
    Write clean, enterprise-level Python code that solves the following task.
    Always follow PEP8, typing annotations, and the Zen of Python.

    Additional style guidelines to follow:
    {style_text}

    Requirements:
    - Include type hints for all function signatures.
    - Add a clear docstring for each function.
    - Import all required types from the typing module explicitly (e.g., Optional, List, Dict).
    - Do NOT include any example calls, print statements, or main blocks.
    - The output must contain ONLY the Python functions/classes with necessary imports.

    Return ONLY the Python code in a markdown block.

    Task: {{task}}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"task": task})
    return clean_code(raw_output)


def generate_tests(code: str, llm: ChatOpenAI) -> str:
    """Generate a pytest suite for the given code, forcing correct imports and realistic error tests."""
    func_names = list_exported_functions(code)
    exceptions = list_raised_exceptions(code)

    func_list_str = ", ".join(func_names) if func_names else "*"
    exception_list_str = ", ".join(exceptions) if exceptions else "none"

    # We let LLM write ONLY test functions/fixtures (no imports).
    test_prompt = PromptTemplate.from_template("""
You are an expert Python developer.
Write a complete pytest test suite for the code below.

Constraints:
- Do NOT include any import statements (no 'import pytest', no 'from ... import ...').
- Assume the following functions are already imported and available in the test namespace: {func_list}.
- Cover normal cases and edge cases.
- Only include error-case tests for these exception types if the code actually raises them: {exception_list}.
  If 'none', do NOT include any tests expecting exceptions.
- Use plain pytest style (no unittest).
- Return ONLY valid Python test code (no markdown fences).

Code under test:
{code}
""")

    test_chain = test_prompt | llm | StrOutputParser()
    raw_tests = test_chain.invoke({
        "func_list": func_list_str,
        "exception_list": exception_list_str,
        "code": code,
    })
    tests_body = clean_code(raw_tests)

    # We prepend our own, correct imports header.
    header_lines = ["import pytest"]
    if func_names:
        header_lines.append(f"from generated_code import {func_list_str}")
    else:
        header_lines.append("from generated_code import *")
    header = "\n".join(header_lines) + "\n\n"

    return header + tests_body


if __name__ == "__main__":
    task = input("Enter task description: ")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Generate code
    code = generate_code(task, llm)

    print("\n=== Generated Code ===\n")
    print(code)

    # Validate code before saving
    if validate_code(code):
        filename = "generated_code.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"\n✅ Code saved to file: {filename}")

        # Generate test suite (imports header is enforced programmatically)
        test_code = generate_tests(code, llm)

        os.makedirs("tests", exist_ok=True)
        test_filename = os.path.join("tests", "test_generated_code.py")
        with open(test_filename, "w", encoding="utf-8") as f:
            f.write(test_code)

        print(f"✅ Test suite generated: {test_filename}")

    else:
        print("\n⚠️ Code was not saved due to syntax errors.")
