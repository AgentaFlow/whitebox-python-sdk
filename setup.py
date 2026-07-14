"""
WhiteBoxAI Python SDK - ML Monitoring and Observability
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    filepath = this_directory / filename
    if not filepath.exists():
        return []
    with open(filepath, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-r")
        ]


# Core requirements
install_requires = [
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "numpy>=1.24.0",
]

# Optional dependencies for integrations
extras_require = {
    "sklearn": ["scikit-learn>=1.3.0"],
    "pytorch": ["torch>=2.0.0"],
    "tensorflow": ["tensorflow>=2.13.0"],
    "transformers": ["transformers>=4.30.0"],
    "langchain": ["langchain>=0.0.200"],
    "boosting": ["xgboost>=1.7.0", "lightgbm>=4.0.0"],
    "all": [
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "tensorflow>=2.13.0",
        "transformers>=4.30.0",
        "langchain>=0.0.200",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "black>=23.11.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "pylint>=3.0.0",
        "mypy>=1.7.0",
        "bandit>=1.7.5",
    ],
}

setup(
    name="whiteboxai-sdk",
    version="0.1.0",
    author="AgentaFlow",
    author_email="support@agentaflow.com",
    description="Official Python SDK for WhiteBoxAI ML monitoring and observability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AgentaFlow/whitebox-python-sdk",
    project_urls={
        "Bug Reports": "https://github.com/AgentaFlow/whitebox-python-sdk/issues",
        "Source": "https://github.com/AgentaFlow/whitebox-python-sdk",
        "Documentation": "https://whitebox.agentaflow.com",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
)
