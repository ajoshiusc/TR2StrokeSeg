"""Setup script for TR2StrokeSeg package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tr2strokeseg",
    version="0.1.0",
    author="Anand Joshi",
    author_email="ajoshi@usc.edu",
    description="Stroke lesion segmentation using nn-UNet trained on Atlas2 dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajoshiusc/TR2StrokeSeg",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tr2stroke-prepare=src.data_preparation.prepare_atlas2:main",
            "tr2stroke-train=src.training.train_nnunet:main",
            "tr2stroke-predict=src.inference.predict:main",
            "tr2stroke-evaluate=src.inference.evaluate:main",
        ],
    },
)
