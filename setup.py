# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="longfact",  # Replace with your preferred package name
    version="0.1.1",
    author="",
    author_email="your.email@example.com",
    description="A short description of your package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/epinnock/long-form-factuality",
    packages=find_packages(include=["*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",   
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    package_data={
    'third_party.factscore': ['demos/*.json', 'demos/**/*.json', 'demos/*', 'demos/**/*'],
    'third_party.factscore.demos': ['*.json'] 
    },
    include_package_data=True,
)