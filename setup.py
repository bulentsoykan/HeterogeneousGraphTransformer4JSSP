from setuptools import setup, find_packages

setup(
    name="hgt-scheduler",
    version="0.1.0",
    description="Exploiting Edge Semantics in Job Shop Scheduling Problem with Heterogeneous Graph Transformers",
    author="Bulent Soykan",
    author_email="Bulent.Soykan@ucf.edu",
    url="https://github.com/bsoykan/HeterogeneousGraphTransformer4JSSP",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "gymnasium>=0.28.0",
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
