import os
import re
import subprocess
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from main import generate_code, generate_tests, validate_code
from style_knowledge_base import add_documents, retrieve_style

# ---------- Constants ----------
CODE_FILENAME = "generated_code_agent.py"
TESTS_DIR = "tests"
TEST_FILENAME = os.path.join(TESTS_DIR, "test_generated_code_agent.py")

# ---------- Helpers ----------

def clean_code(text: str) -> str:
    """Remove markdown fences from LLM output."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def save_file(filename: str, content: str) -> None:
    """Save text to a file, ensuring directory exists if needed."""
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def run_pytest(test_file: str) -> tuple[bool, str]:
    """Run pytest on the given test file and return success flag and output."""
    result = subprocess.run(
        ["pytest", test_file, "-v", "--maxfail=1", "--disable-warnings"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


# ---------- Workflow ----------

def auto_generate_and_test(task: str) -> None:
    """End-to-end pipeline: generate code, generate tests, run pytest, repair if needed."""

    llm_code = ChatOpenAI(model="gpt-5-mini", temperature=0)
    llm_tests = ChatOpenAI(model="gpt-5-mini", temperature=0)
    llm_repair = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 1. Generate code (with RAG style guidance)
    print("📝 Generating code...")
    add_documents()
    style_guidelines = retrieve_style(task)
    style_text = "\n".join([doc.page_content for doc in style_guidelines])

    code = generate_code(task, llm_code)
    code = clean_code(code)
    if not validate_code(code):
        print("❌ Generated code is invalid Python. Aborting.")
        return

    save_file(CODE_FILENAME, code)
    print(f"✅ Code saved to {CODE_FILENAME}")

    # 2. Generate tests
    print("🧪 Generating tests...")
    raw_tests = generate_tests(code, llm_tests)
    tests = clean_code(raw_tests)
    tests = tests.replace("from generated_code import", "from generated_code_agent import")
    save_file(TEST_FILENAME, tests)
    print(f"✅ Tests saved to {TEST_FILENAME}")

    # 3. Run pytest loop with auto-repair
    max_rounds = 3
    for round_no in range(1, max_rounds + 1):
        print(f"\n🚀 Running pytest (round {round_no})...")
        success, output = run_pytest(TEST_FILENAME)

        if success:
            print("🎉 All tests passed successfully!\n")
            print(output)  # pokaż pełny wynik pytest
            return
        else:
            print("❌ Tests failed. Sending to LLM for repair...")
            # 4. Repair code and tests
            repair_prompt = f"""
The following Python code and tests failed pytest.

--- CODE ---
{code}

--- TESTS ---
{tests}

--- PYTEST OUTPUT ---
{output}

Fix the code and/or tests so that pytest passes.
Return only valid Python code for the fixed code first,
then below a marker line '### TESTS ###',
return valid pytest tests.
"""
            repaired = llm_repair.invoke(repair_prompt).content
            # Split code and tests
            if "### TESTS ###" in repaired:
                new_code, new_tests = repaired.split("### TESTS ###", 1)
                code = clean_code(new_code)
                tests = clean_code(new_tests)
                tests = tests.replace("from generated_code import", "from generated_code_agent import")
                save_file(CODE_FILENAME, code)
                save_file(TEST_FILENAME, tests)
                print("🔧 Repaired code and tests written.")
            else:
                print("⚠️ Repair step failed to produce code/tests.")
                break

    print("🚨 Could not fix code/tests after retries.")


def main():
    load_dotenv()
    print("🤖 Auto-Agent ready. Describe the Python task to implement:")
    task = input("> ").strip()
    auto_generate_and_test(task)


if __name__ == "__main__":
    main()
