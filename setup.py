import os

from setuptools import setup, find_packages

setup(
    name="cache-rl-optimization",
    version="0.1.0",
    description="Cache Optimization using Reinforcement Learning",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="sdu-f25-mjm",
    author_email="",
    url="https://github.com/sdu-f25-mjm/ers-ml-trainer",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.115.12",
        "uvicorn>=0.34.0",
        "starlette>=0.46.1",
        "pydantic>=2.11.3",
        "sqlalchemy>=2.0.21",
        "gymnasium>=1.1.1",
        "psycopg2-binary>=2.9.10"
        "stable-baselines3>=2.6.0",
        "numpy<2.2.0",
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "sqlalchemy>=2.0.40",
        "python-dotenv>=1.1.0",
        "typing-extensions>=4.13.2",
        "python-multipart>=0.0.20",
        "mysql-connector-python>=9.2.0",
        "psutil>=7.0.0",
        "tensorflow_cpu>=2.19.0",
    ],
    entry_points={
        "console_scripts": [
            "cache-rl-api=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Caching",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)
