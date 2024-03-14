from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fastfizenv',
    version='0.0.1',
    packages=find_packages(),
    license='MIT',
    install_requires=requirements,
    py_modules=['src'],
)
