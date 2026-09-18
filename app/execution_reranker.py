import subprocess
import tempfile
import os
import re
import logging

def parse_query_for_examples(text: str) -> list[dict]:
    examples = []
    examples_section = re.search(r'-----\s*Examples\s*-----(.*?)(?:-----\s*Note\s*-----|$)', text, re.IGNORECASE | re.DOTALL)
    if examples_section:
        ex_text = examples_section.group(1).strip()
        blocks = re.findall(r'Input\s*(.*?)\s*Output\s*(.*?)(?=\nInput\s*|$)', ex_text, re.IGNORECASE | re.DOTALL)
        for inp, outp in blocks:
            examples.append({
                "input": inp.strip(),
                "output": outp.strip()
            })
    return examples

def run_code_with_timeout(code: str, stdin: str, timeout: int = 2) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "solution.py")
        with open(filepath, "w") as f:
            f.write(code)
            
        try:
            result = subprocess.run(
                [os.sys.executable, filepath],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
            
def fuzzy_match(actual: str, expected: str) -> bool:
    def clean(s):
        return re.sub(r'\s+', '', str(s)).strip()
    return clean(actual) == clean(expected)

def verify_candidate(code: str, examples: list[dict], timeout=2) -> bool:
    if not examples:
        return False
        
    for ex in examples:
        success, stdout = run_code_with_timeout(code, ex["input"], timeout=timeout)
        if not success:
            return False
        if not fuzzy_match(stdout, ex["output"]):
            return False
    return True
