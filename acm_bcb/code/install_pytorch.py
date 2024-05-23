import subprocess
import sys

def check_dependencies():
    try:
        # Check for dependency issues after a dry-run installation
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print("Dependency issues found:")
            print(result.stdout)
            return False
        return True
    except Exception as e:
        print(f"An error occurred while checking dependencies: {e}")
        return False

def install_pytorch():
    try:
        # Attempt to install PyTorch in dry-run mode
        print("Checking for compatibility issues before installation...")
        dry_run_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if dry_run_result.returncode != 0:
            print("Compatibility issues found during dry-run installation:")
            print(dry_run_result.stdout)
            return

        print("No compatibility issues found. Proceeding with installation...")
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if install_result.returncode == 0:
            print("PyTorch installed successfully.")
            if check_dependencies():
                print("No dependency issues found after installation.")
            else:
                print("Dependency issues found after installation.")
        else:
            print("Installation failed:")
            print(install_result.stdout)

    except Exception as e:
        print(f"An error occurred during the installation process: {e}")

if __name__ == "__main__":
    install_pytorch()
