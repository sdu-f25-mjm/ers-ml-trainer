import os

from setuptools import setup, find_packages

setup(
    name="cache-rl-optimization",
    version="0.1.0",
    description="Cache Optimization using Reinforcement Learning",
    long_description=open("docs/README.md").read() if os.path.exists("docs/README.md") else "",
    long_description_content_type="text/markdown",
    author="sdu-f25-mjm",
    author_email="",
    url="https://github.com/sdu-f25-mjm/ers-ml-trainer",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi==0.103.1",
        "uvicorn",
        "pydantic==1.10.8",
        "sqlalchemy==2.0.21",
        "torch>=2.0.0",
        "gymnasium>=0.28.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.5.0",
        "python-multipart>=0.0.6",
        "mysql-connector-python==8.0.33",
        "psutil>=5.9.0",
        "tensorflow>=2.14.0",
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