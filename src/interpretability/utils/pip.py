# stdlib
from pathlib import Path
import subprocess
import sys

current_dir = Path(__file__).parent

predefined = {
    "shap": "shap>=0.41.0",
    "combo": "git+https://github.com/yzhao062/combo",
    "symbolic_pursuit": "git+https://github.com/vanderschaarlab/Symbolic-Pursuit",
}


def install(packages: list) -> None:
    for package in packages:
        install_pack = package
        if package in predefined:
            install_pack = predefined[package]
        print(f"Installing {install_pack}")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", install_pack],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except BaseException as e:
            print(f"failed to install package {package}: {e}")
