from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="seminartools",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    author="Maurits, Philipp, Matthias, Stefan",
)
