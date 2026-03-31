import os
import requests
import yaml
import sys

def check_openenv_yaml():
    if not os.path.exists("openenv.yaml"):
        return False, "openenv.yaml is missing"
    with open("openenv.yaml", "r") as f:
        data = yaml.safe_load(f)
        if "spec_version" not in data:
            return False, "spec_version missing from openenv.yaml"
    return True, "openenv.yaml looks good"

def check_inference_script():
    if not os.path.exists("inference.py"):
        return False, "inference.py is missing from root directory"
    with open("inference.py", "r") as f:
        content = f.read()
        if "API_BASE_URL" not in content or "MODEL_NAME" not in content or "HF_TOKEN" not in content:
            return False, "inference.py doesn't use required environment variables"
        if "OpenAI" not in content:
            return False, "inference.py doesn't use the OpenAI Client"
    return True, "inference.py looks good"

def run_tests():
    passed = True
    print("--- Running Validations ---")
    tests = [
        check_openenv_yaml,
        check_inference_script,
    ]
    for test in tests:
        res, msg = test()
        print(f"[{'PASS' if res else 'FAIL'}] {msg}")
        if not res:
            passed = False
            
    print("---------------------------")
    if passed:
        print("Static validation PASSED.")
        sys.exit(0)
    else:
        print("Static validation FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
