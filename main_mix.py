import os
import re
import ast
import subprocess
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


def run_pytest() -> tuple[bool, str]:
    """Run pytest and return (success, output)."""
    result = subprocess.run(
        ["pytest", "-q", "--tb=short"],
        capture_output=True,
        text=True
    )
    success = result.returncode == 0
    return success, result.stdout + result.stderr


def generate_code(task: str, llm_code: ChatOpenAI) -> str:
    """Generate Python code from task description with style context."""
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
    chain = prompt | llm_code | StrOutputParser()
    raw_output = chain.invoke({"task": task})
    return clean_code(raw_output)


def generate_tests(code: str, llm_tests: ChatOpenAI) -> str:
    """Generate pytest suite for the given code (without imports)."""
    test_prompt = PromptTemplate.from_template("""
    You are an expert Python developer.
    Write a pytest test suite for the following code.

    Constraints:
    - Do NOT include any import statements.
    - Only write test functions.
    - Cover normal cases, edge cases, and (if relevant) error cases.

    Code under test:
    {code}
    """)

    test_chain = test_prompt | llm_tests | StrOutputParser()
    raw_tests = test_chain.invoke({"code": code})
    tests_body = clean_code(raw_tests)

    # Doklejamy poprawny nagłówek sami
    header = "import pytest\nfrom generated_code import *\n\n"
    return header + tests_body


def repair_code(code: str, tests: str, errors: str, llm_repair: ChatOpenAI) -> str:
    """Ask LLM to repair broken code based on failing tests and traceback."""
    repair_prompt = PromptTemplate.from_template("""
    You are an expert Python developer.
    The following code failed some pytest tests.

    Code:
    {code}

    Tests:
    {tests}

    Pytest output (errors and failures):
    {errors}

    Task:
    - Fix the code so that all tests pass.
    - Do not modify the tests.
    - Return ONLY the corrected Python code in a markdown block.
    """)

    repair_chain = repair_prompt | llm_repair | StrOutputParser()
    raw_fixed = repair_chain.invoke({"code": code, "tests": tests, "errors": errors})
    return clean_code(raw_fixed)


if __name__ == "__main__":
    task = input("Enter task description: ")

    # Modele (dual-mode setup)
    llm_code = ChatOpenAI(model="gpt-5-mini", temperature=0)   # generowanie kodu
    llm_tests = ChatOpenAI(model="gpt-5-mini", temperature=0)   # generowanie testów
    llm_repair = ChatOpenAI(model="gpt-5-mini", temperature=0)  # poprawki kodu

    # Step 1: generate code
    code = generate_code(task, llm_code)
    print("\n=== Generated Code ===\n")
    print(code)

    if not validate_code(code):
        print("\n⚠️ Invalid Python code, exiting.")
        exit(1)

    with open("generated_code_mix.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("\n✅ Code saved to generated_code_mix.py")

    # Step 2: generate tests
    test_code = generate_tests(code, llm_tests)
    os.makedirs("tests", exist_ok=True)
    with open("tests/test_generated_code_mix.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    print("✅ Test suite saved to tests/test_generated_code_mix.py")

    # Step 3: run pytest
    success, output = run_pytest()
    print("\n=== Pytest Output ===")
    print(output)

    # Step 4: if tests fail, repair
    if not success:
        print("⚠️ Tests failed. Attempting to repair code...")

        fixed_code = repair_code(code, test_code, output, llm_repair)
        print("\n=== Fixed Code ===\n")
        print(fixed_code)

        if validate_code(fixed_code):
            with open("generated_code_mix.py", "w", encoding="utf-8") as f:
                f.write(fixed_code)
            print("\n✅ Fixed code saved to generated_code_mix.py")

            # Run pytest again
            success, output = run_pytest()
            print("\n=== Pytest Output After Repair ===")
            print(output)

            if success:
                print("🎉 All tests passed after repair!")
            else:
                print("❌ Tests still failing, manual fix needed.")
        else:
            print("❌ Fixed code is not valid Python.")
    else:
        print("🎉 All tests passed on first run!")
