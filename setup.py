from pathlib import Path

from setuptools import setup

readme_path = Path(__file__).absolute().parent.joinpath('README.md')

with readme_path.open('r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='cinnamon_th',
    version='0.1',
    author='Federico Ruggeri',
    author_email='federico.ruggeri6@unibo.it',
    description='[Torch Package] A simple high-level framework for research',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/federicoruggeri/cinnamon_th',
    project_urls={
        'Bug Tracker': "https://github.com/federicoruggeri/cinnamon_th/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    license='MIT',
    packages=['cinnamon_th',
              'cinnamon_th.components',
              'cinnamon_th.configurations',
              ],
    python_requires=">=3.6"
)
