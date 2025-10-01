# -*- coding: utf-8 -*-
import os
import subprocess
from dotenv import load_dotenv
# Import ChatOpenAI and tool decorator from LangChain
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Load environment variables (e.g., API keys from .env file)
load_dotenv()

# Utility functions for code generation, testing, and repair
def generate_code(task: str, llm) -> str:
    """Use the LLM to generate Python code for the given task description."""
    prompt = (
        "You are an expert Python developer tasked with writing code to accomplish the following task:\n"
        f"{task}\n\n"
        "Requirements:\n"
        "- Provide the complete Python implementation (functions, classes, etc.) that fulfills the task.\n"
        "- Include appropriate function/class docstrings and comments.\n"
        "- Implement error handling for edge cases: if input types are incorrect, raise TypeError; if input values are invalid (e.g., empty or malformed lists), raise ValueError.\n"
        "- Do not include any explanation or commentary, only output the code."
    )
    # Get response from the language model
    response = llm.predict(prompt)
    return response.strip()

def generate_tests(prompt: str, llm) -> str:
    """Use the LLM to generate a pytest test suite based on the given prompt."""
    # Get response from the language model for test generation
    response = llm.predict(prompt)
    return response.strip()

def repair_code(code: str, tests: str, errors: str, llm) -> str:
    """Use the LLM to repair the code based on test failures and error output."""
    prompt = (
        "You are an expert Python developer and code fixer.\n"
        "The following Python code has failing tests.\n\n"
        "Code:\n" + code + "\n\n"
        "Tests:\n" + tests + "\n\n"
        "Test results (failures and errors):\n" + errors + "\n\n"
        "Please analyze the failures and modify the code to fix the issues.\n"
        "Do not alter the tests. Output only the corrected code, without any additional commentary."
    )
    # Get the fixed code from the language model
    response = llm.predict(prompt)
    return response.strip()

def run_pytest():
    """Run pytest on the generated code and capture the result.
    Returns a tuple (success: bool, output: str)."""
    try:
        # Run pytest quietly (no detailed output except summary and errors)
        result = subprocess.run(["pytest", "-q"], capture_output=True, text=True)
    except Exception as e:
        return False, f"Pytest execution failed: {e}"
    output = result.stdout + result.stderr
    # success is True if return code is 0 (all tests passed)
    success = (result.returncode == 0)
    return success, output

def validate_code(code: str) -> bool:
    """Check if the generated code is syntactically valid (compilable)."""
    try:
        compile(code, "generated_code_A_mix.py", "exec")
        return True
    except Exception:
        return False

# === Tools ===

@tool
def GenerateCodeTool(task: str) -> str:
    """Generate Python code for the given task."""
    # Initialize LLM for code generation (deterministic output)
    llm_code = ChatOpenAI(model="gpt-5-mini", temperature=0)
    code = generate_code(task, llm_code)
    # Save the generated code to file
    with open("generated_code_A_mix.py", "w", encoding="utf-8") as f:
        f.write(code)
    return code

@tool
def GenerateTestsTool(code: str) -> str:
    """Generate pytest test suite for the given code (only for public functions)."""
    # Initialize LLM for test generation
    llm_tests = ChatOpenAI(model="gpt-5-mini", temperature=0)
    # Construct a prompt with constraints for test generation
    constrained_prompt = f"""You are an expert Python developer.
Write a pytest test suite for the following code.

Constraints:
- Do NOT include any import statements except pytest and the module under test.
- Only test the public API functions (ignore private helpers starting with "_").
- Only write test functions.
- Cover normal cases, edge cases, and error cases.
- Expected exceptions:
    * TypeError → when arguments have the wrong type.
    * ValueError → when arguments are lists but invalid (empty, irregular, etc).

Code under test:
{code}
"""
    tests = generate_tests(constrained_prompt, llm_tests)
    # Ensure the tests directory exists and save the tests to a file
    os.makedirs("tests", exist_ok=True)
    with open("tests/test_generated_code_A_mix.py", "w", encoding="utf-8") as f:
        f.write(tests)
    return tests

@tool
def RunPytestTool(_: str = "") -> str:
    """Run pytest on the generated code and return the output."""
    _, output = run_pytest()
    return output

@tool
def RepairCodeTool(input_data: dict) -> str:
    """Repair the generated code using pytest output.
    Input must contain 'code', 'tests', and 'errors'."""
    # Initialize LLM for code repair
    llm_repair = ChatOpenAI(model="gpt-5-mini", temperature=0)
    fixed_code = repair_code(
        input_data.get("code", ""),
        input_data.get("tests", ""),
        input_data.get("errors", ""),
        llm_repair
    )
    # Save the repaired code back to the code file
    with open("generated_code_A_mix.py", "w", encoding="utf-8") as f:
        f.write(fixed_code)
    return fixed_code

# === Pipeline ===

def main():
    print("🤖 Auto-Agent ready. Describe the task you want to implement in Python:")
    task = input("> ")

    # Step 1: Generate code
    code = GenerateCodeTool.run(task)
    print("\n✅ Code generated and saved to generated_code_A_mix.py")

    # Step 2: Validate code
    if not validate_code(code):
        print("❌ Generated code is invalid. Exiting.")
        return

    # Step 3: Generate tests
    tests = GenerateTestsTool.run(code)
    print("✅ Tests generated and saved to tests/test_generated_code_A_mix.py")

    # Step 4: Run pytest
    print("\n=== Running pytest ===")
    output = RunPytestTool.run("")
    print(output)

    # Step 5: If tests fail, attempt repair
    if "FAILED" in output or "Error" in output:
        print("⚠️ Tests failed. Attempting repair...")
        fixed_code = RepairCodeTool.invoke({
            "args": {
                "code": code,
                "tests": tests,
                "errors": output
            }
        })
        print("\n✅ Fixed code saved to generated_code_A_mix.py")

        # Run tests again after repair
        print("\n=== Running pytest after repair ===")
        output2 = RunPytestTool.run("")
        print(output2)
        if "FAILED" not in output2 and "Error" not in output2:
            print("🎉 All tests passed after repair!")
        else:
            print("❌ Tests still failing. Manual review needed.")
    else:
        print("🎉 All tests passed on first run!")

if __name__ == "__main__":
    main()
