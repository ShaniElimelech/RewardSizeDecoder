import subprocess
import sys

packages = [
    "scikit-learn",
    "numpy",
    "matplotlib",
    "pandas",
    "optuna",
    "datajoint",
    "seaborn",
    "opencv-python",
    "imbalanced-learn",
    "scipy"
]

def main():
    for pkg in packages:
        print(f" Installing or upgrading {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg])

if __name__ == "__main__":
    main()
