import subprocess
import sys

def run_command(url: str):
    try:
        print(f"Running main.py for: {url}")

        result = subprocess.run(
            ["python", "codegraph/main.py", url],
            check=True,
            capture_output=True,
            text=True
        )

        print("=== main.py output ===")
        print(result.stdout)

        print("Running predict.py...")

        result = subprocess.run([sys.executable, "Node_Classification/predict.py", "data/output.json"]
)

        print("=== predict.py output ===")
        print(result.stdout)

        result = subprocess.run([sys.executable, "Label_Classification/infer_codebert.py", url])
        print(result.stdout)

        print("Pipeline completed successfully.")

    except subprocess.CalledProcessError as e:
        print("\n========== ERROR ==========")
        print(f"Command failed: {e.cmd}")
        print(f"Return code: {e.returncode}")

        if e.stdout:
            print("\n--- STDOUT ---")
            print(e.stdout)

        if e.stderr:
            print("\n--- STDERR ---")
            print(e.stderr)

        print("===========================\n")


path=input("Enter the GitHub repository URL: ")
run_command(path)


subprocess.run(
    ["npm", "run", "dev"],
    cwd="/Users/ankushpal/Desktop/Combine/code-graph"
)