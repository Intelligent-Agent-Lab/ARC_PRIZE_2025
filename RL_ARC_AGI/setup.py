from setuptools import setup, find_packages

setup(
    name='rl_arc_agi',
    version='0.1.0',
    description='ARC AGI 2 Reinforcement Learning Environment',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'matplotlib',
    ],
    python_requires='>=3.7',
)