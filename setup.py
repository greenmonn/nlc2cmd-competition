from setuptools import setup, find_packages


setup(
    name='nl2bash',
    version='1.0.0',
    author_email='greenmon@kaist.ac.kr',
    description='Modified for NLC2CMD Competition',
    packages=find_packages(),
    install_requires=[
        "nltk==3.4.5",
        "tqdm>=4.9.0",
        "nose>=1.1.2",
        "numpy>=1.7",
        "scipy>0.13.3",
        "six>=1.8",
        "matplotlib"
    ],
)
