import os
import re
import ast
import subprocess
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from style_knowledge_base_advanced import add_documents, retrieve_style

# Load environment variables
load_dotenv()

CODE_FILENAME = "generated_code_agent_advanced.py"
TESTS_DIR = "tests"
TEST_FILENAME = os.path.join(TESTS_DIR, "test_generated_code_agent_advanced.py")


def clean_code(text: str) -> str:
    """Remove markdown code fences (```python ... ```)."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate_code(code: str) -> bool:
    """Check if generated code is valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in generated code: {e}")
        return False


def run_pytest() -> tuple[bool, str]:
    """Run pytest ONLY on our generated test file."""
    result = subprocess.run(
        ["pytest", "-q", "--tb=short", TEST_FILENAME],
        capture_output=True,
        text=True
    )
    success = result.returncode == 0
    return success, result.stdout + result.stderr


def generate_code(task: str, llm: ChatOpenAI) -> str:
    """Generate Python code using RAG style guidelines."""
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
    - Import required types from typing explicitly (List, Dict, Optional, etc.).
    - Do NOT include any example calls, print statements, or main blocks.
    - The output must contain ONLY Python code.

    Return ONLY the Python code in a markdown block.

    Task: {{task}}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"task": task})
    return clean_code(raw_output)


def generate_tests(code: str, llm: ChatOpenAI) -> str:
    """Generate pytest test suite for the given code, with fixed import header."""
    test_prompt = PromptTemplate.from_template("""
    You are an expert Python developer.
    Write a pytest test suite for the following code.

    Constraints:
    - Do NOT include any import statements.
    - Only write test functions.
    - Cover normal cases, edge cases, and error cases.

    Code under test:
    {code}
    """)

    test_chain = test_prompt | llm | StrOutputParser()
    raw_tests = test_chain.invoke({"code": code})
    tests_body = clean_code(raw_tests)

    # Always prepend our correct imports
    header = f"import pytest\nfrom {CODE_FILENAME[:-3]} import *\n\n"
    return header + tests_body


def repair_code(code: str, tests: str, errors: str, llm: ChatOpenAI) -> str:
    """Repair broken code based on failing pytest output."""
    repair_prompt = PromptTemplate.from_template("""
    You are an expert Python developer.
    The following code failed pytest tests.

    Code:
    {code}

    Tests:
    {tests}

    Pytest output (errors and failures):
    {errors}

    Task:
    - Fix ONLY the code so that all tests pass.
    - Do not modify the tests.
    - Return ONLY the corrected Python code in a markdown block.
    """)

    repair_chain = repair_prompt | llm | StrOutputParser()
    raw_fixed = repair_chain.invoke({"code": code, "tests": tests, "errors": errors})
    return clean_code(raw_fixed)


if __name__ == "__main__":
    task = input("Enter task description: ")

    # Use separate models for different steps
    llm_code = ChatOpenAI(model="gpt-5-mini", temperature=0)
    llm_tests = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # can swap to gpt-5-mini if you want
    llm_repair = ChatOpenAI(model="gpt-5-mini", temperature=0)

    # Step 1: generate code
    code = generate_code(task, llm_code)
    print("\n=== Generated Code ===\n")
    print(code)

    if not validate_code(code):
        print("⚠️ Invalid Python code, exiting.")
        exit(1)

    with open(CODE_FILENAME, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"✅ Code saved to {CODE_FILENAME}")

    # Step 2: generate tests
    test_code = generate_tests(code, llm_tests)
    os.makedirs(TESTS_DIR, exist_ok=True)
    with open(TEST_FILENAME, "w", encoding="utf-8") as f:
        f.write(test_code)
    print(f"✅ Test suite saved to {TEST_FILENAME}")

    # Step 3: run pytest
    success, output = run_pytest()
    print("\n=== Pytest Output ===")
    print(output)

    # Step 4: repair if needed
    if not success:
        print("⚠️ Tests failed. Attempting to repair code...")

        fixed_code = repair_code(code, test_code, output, llm_repair)
        print("\n=== Fixed Code ===\n")
        print(fixed_code)

        if validate_code(fixed_code):
            with open(CODE_FILENAME, "w", encoding="utf-8") as f:
                f.write(fixed_code)
            print(f"✅ Fixed code saved to {CODE_FILENAME}")

            # Run pytest again
            success, output = run_pytest()
            print("\n=== Pytest Output After Repair ===")
            print(output)

            if success:
                print("🎉 All tests passed after repair!")
            else:
                print("❌ Tests still failing, manual fix required.")
        else:
            print("❌ Fixed code is not valid Python.")
    else:
        print("🎉 All tests passed on first run!")
