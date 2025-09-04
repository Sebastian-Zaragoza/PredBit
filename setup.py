from setuptools import setup, find_packages

setup(
    name="predbit_ml",
    version="0.1.0",
    author="Sebastian Galindo Zaragoza",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "pyyaml",
        "joblib",
        "xgboost"
    ],
    python_requires=">=3.10"
)