from setuptools import find_packages, setup

setup(
    name="AttentionMechanism",
    version="1.0.2",
    author="gongyan",
    author_email="gongyan2020@foxmail.com",
    description=(
        "This repository contains an implementation of many attention mechanism models."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords=(
        "Attention",
        "Machine Learning",
        "Deep Learning",
        "Neural Networks",
        "Pytorch"
    ),
    license="Apache",
    url="https://github.com/gongyan1/Attention-Mechanism-Pytorch",
    # package_dir={"": "."},
    packages=find_packages(),
    python_requires=">=3.7.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)