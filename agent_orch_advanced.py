import os
import subprocess
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from main_advanced import generate_code, generate_tests, repair_code, validate_code

# ---------- Constants ----------
CODE_FILENAME = "generated_code_agent_advanced.py"
TESTS_DIR = "./tests"
TEST_FILENAME = os.path.join(TESTS_DIR, "test_generated_code_agent_advanced.py")


# ---------- Helpers ----------
def save_file(filename: str, content: str):
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def run_pytest() -> tuple[bool, str]:
    """Run pytest ONLY on the generated test file."""
    result = subprocess.run(
        ["pytest", "-q", "--tb=short", TEST_FILENAME],
        capture_output=True,
        text=True
    )
    success = result.returncode == 0
    return success, result.stdout + result.stderr


# ---------- Orchestration ----------
def main():
    load_dotenv()
    print("🤖 Auto-Agent Advanced ready. Describe the Python task to implement:")
    task = input("> ").strip()

    # Models
    llm_code = ChatOpenAI(model="gpt-5-mini", temperature=0)   # main code
    llm_tests = ChatOpenAI(model="gpt-4o-mini", temperature=0) # test generation
    llm_repair = ChatOpenAI(model="gpt-5-mini", temperature=0) # repair if needed

    # Step 1: Generate code
    print("📝 Generating code...")
    code = generate_code(task, llm_code)
    save_file(CODE_FILENAME, code)
    print(f"✅ Code saved to {CODE_FILENAME}")

    if not validate_code(code):
        print("❌ Invalid Python code. Exiting.")
        return

    # Step 2: Generate tests
    print("🧪 Generating tests...")
    os.makedirs(TESTS_DIR, exist_ok=True)
    raw_tests = generate_tests(code, llm_tests)

    # Fix imports in tests
    module_name = os.path.splitext(os.path.basename(CODE_FILENAME))[0]
    fixed_tests = raw_tests.replace("from generated_code import", f"from {module_name} import")
    save_file(TEST_FILENAME, fixed_tests)
    print(f"✅ Test suite saved to {TEST_FILENAME}")

    # Step 3: Run pytest
    print("\n=== Running Pytest ===")
    success, output = run_pytest()
    print(output)

    # Step 4: If tests fail, repair code
    if not success:
        print("⚠️ Tests failed. Attempting to repair code...")
        fixed_code = repair_code(code, fixed_tests, output, llm_repair)
        save_file(CODE_FILENAME, fixed_code)
        print("✅ Fixed code saved. Re-running tests...\n")

        if validate_code(fixed_code):
            success, output = run_pytest()
            print(output)
            if success:
                print("🎉 All tests passed after repair!")
            else:
                print("❌ Tests still failing, manual fix needed.")
        else:
            print("❌ Fixed code is not valid Python.")
    else:
        print("🎉 All tests passed on first run!")


if __name__ == "__main__":
    main()
