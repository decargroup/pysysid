"""``pysysid`` setup file."""

import setuptools

with open('README.rst', 'r') as f:
    readme = f.read()

setuptools.setup(
    name='pysysid',
    version='0.1.0',
    description=('System identification library in Python, compatible with '
                 '`scikit-learn`'),
    long_description=readme,
    author='Steven Dahdah',
    author_email='steven.dahdah@mail.mcgill.ca',
    url='https://github.com/decargroup/pysysid',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
    project_urls={
        'Documentation': 'https://pysysid.readthedocs.io/en/stable',
        'Source': 'https://github.com/decargroup/pysysid',
        'Tracker': 'https://github.com/decargroup/pysysid/issues',
        'PyPI': 'https://pypi.org/project/pysysid/',
        # 'DOI': 'https://doi.org/10.5281/zenodo.5576490',
    },
    packages=setuptools.find_packages(exclude=('tests', 'examples', 'doc')),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.1',
        'matplotlib>=3.5.1',
    ],
)
