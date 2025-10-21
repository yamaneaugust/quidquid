"""Setup script for lesion pre-screening system."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lesion-prescreening",
    version="0.1.0",
    author="CNN Lesion Screening Team",
    description="CNN-based lesion pre-screening and documentation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quidquid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lesion-train=src.model.train:main",
            "lesion-predict=src.inference.predict:main",
        ],
    },
)
