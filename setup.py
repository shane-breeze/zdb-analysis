import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="zdb-analysis",
    version="0.1.7",
    author="Shane Breeze",
    author_email="sdb15@ic.ac.uk",
    scripts=[
        "zdb/scripts/zdb_analyse.py",
        "zdb/scripts/zdb_resume.py",
        "zdb/scripts/zdb_skim.py",
    ],
    description="Pandas + Z inv analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        "Development Status :: 3 - Alpha",
    ),
)
