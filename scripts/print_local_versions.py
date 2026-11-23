# Run this in the same kernel that produced the final Notebook 05 results, then
# paste the output into requirements.txt. Parity depends on it.
#
# From a notebook cell:
#     %run scripts/print_local_versions.py
# From that kernel's shell:
#     python scripts/print_local_versions.py
#
# The first printed line gives the Python major.minor for the Dockerfile's
# PYTHON_VERSION build argument. The remaining lines replace the
# REPLACE_WITH_LOCAL_VERSION placeholders in requirements.txt, in order.

import importlib
import platform

version = platform.python_version()
print(f"# python {version} on {platform.machine()}")
print(f"# Dockerfile: --build-arg PYTHON_VERSION={'.'.join(version.split('.')[:2])}")

for package, module in [("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"),
                        ("scikit-learn", "sklearn"), ("xgboost", "xgboost"),
                        ("joblib", "joblib"), ("pyarrow", "pyarrow")]:
    try:
        print(f"{package}=={importlib.import_module(module).__version__}")
    except ImportError:
        print(f"# {package} not installed")
