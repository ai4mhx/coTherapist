"""Setup configuration for coTherapist."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="coTherapist",
    version="0.1.0",
    author="ai4mhx",
    description="A Mental Healthcare AI Copilot - Domain-specific LLM for therapeutic conversations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai4mhx/coTherapist",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cotherapist-train=scripts.train:main",
            "cotherapist-evaluate=scripts.evaluate:main",
            "cotherapist-chat=scripts.chat:main",
        ],
    },
)
