from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
requirements = (
        (here / "requirements.txt").read_text("utf8").strip().split("\n")
    )

setup(
    name='pixel-rectified-flow',
    version='1.0.0',
    author='Your Name',
    author_email='your-email@example.com',
    description='A short description of your package',
    packages=find_packages(),
    install_requires=requirements,
)
