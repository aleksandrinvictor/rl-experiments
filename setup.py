from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='rl_experiments',
    # package_dir={'': 'rl_experiments'},
    packages=find_packages(),
    install_requires=requirements,
    version='0.1',
    description='Implementations of RL algortihms',
    author='Aleksandrin Victor',
)
