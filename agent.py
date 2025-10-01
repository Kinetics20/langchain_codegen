import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from main_mix import generate_code, generate_tests, repair_code, run_pytest, validate_code

# Load env
load_dotenv()


# === Tools ===

@tool
def GenerateCodeTool(task: str) -> str:
    """Generate Python code for the given task."""
    llm_code = ChatOpenAI(model="gpt-5-mini", temperature=0)
    code = generate_code(task, llm_code)
    with open("generated_code_A.py", "w", encoding="utf-8") as f:
        f.write(code)
    return code


@tool
def GenerateTestsTool(code: str) -> str:
    """Generate pytest test suite for the given code (only for public functions)."""
    llm_tests = ChatOpenAI(model="gpt-5-mini", temperature=0)

    # More restrictive prompt for test generation
    constrained_prompt = f"""
    You are an expert Python developer.
    Write a pytest test suite for the following code.

    Constraints:
    - Do NOT include any import statements.
    - Only test the public API functions (ignore private helpers starting with "_").
    - Only write test functions.
    - Cover normal cases, edge cases, and (if relevant) error cases.

    Code under test:
    {code}
    """

    tests = generate_tests(constrained_prompt, llm_tests)
    os.makedirs("tests", exist_ok=True)
    with open("tests/test_generated_code_A.py", "w", encoding="utf-8") as f:
        f.write(tests)
    return tests


@tool
def RunPytestTool(_: str = "") -> str:
    """Run pytest on the generated code and return the output."""
    success, output = run_pytest()
    return output


@tool
def RepairCodeTool(args: dict) -> str:
    """Repair the generated code using pytest output.
    Args must contain 'code', 'tests', and 'errors'."""
    llm_repair = ChatOpenAI(model="gpt-5-mini", temperature=0)
    fixed_code = repair_code(args["code"], args["tests"], args["errors"], llm_repair)
    with open("generated_code_A.py", "w", encoding="utf-8") as f:
        f.write(fixed_code)
    return fixed_code


# === Pipeline ===

def main():
    print("🤖 Auto-Agent ready. Describe the task you want to implement in Python:")
    task = input("> ")

    # Step 1: Generate code
    code = GenerateCodeTool.run(task)
    print("\n✅ Code generated and saved to generated_code_A.py")

    # Step 2: Validate code
    if not validate_code(code):
        print("❌ Generated code is invalid. Exiting.")
        return

    # Step 3: Generate tests
    tests = GenerateTestsTool.run(code)
    print("✅ Tests generated and saved to tests/test_generated_code_A.py")

    # Step 4: Run pytest
    print("\n=== Running pytest ===")
    output = RunPytestTool.run("")
    print(output)

    # Step 5: If tests fail, repair
    if "FAILED" in output or "Error" in output:
        print("⚠️ Tests failed. Attempting repair...")
        fixed_code = RepairCodeTool.invoke({
            "args": {
                "code": code,
                "tests": tests,
                "errors": output
            }
        })
        print("\n✅ Fixed code saved to generated_code_A.py")

        # Run pytest again
        print("\n=== Running pytest after repair ===")
        output2 = RunPytestTool.run("")
        print(output2)
        if "FAILED" not in output2:
            print("🎉 All tests passed after repair!")
        else:
            print("❌ Tests still failing. Manual review needed.")
    else:
        print("🎉 All tests passed on first run!")


if __name__ == "__main__":
    main()