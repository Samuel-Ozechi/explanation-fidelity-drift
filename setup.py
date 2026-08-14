from setuptools import find_packages, setup


setup(
    name="explanation-fidelity-drift",
    version="0.1.0",
    description="Explanation fidelity drift experiments",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
    ],
)
