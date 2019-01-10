from setuptools import setup
from version import __version__

# Get the long description from the README file
with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hmm_filter',
    version=__version__,
    author='Michele Dallachiesa',
    author_email='michele.dallachiesa@minodes.com',
    packages=['hmm_filter'],
    scripts=[],
    url='https://github.com/minodes/hmm_filter',
    license='MIT',
    description='Improve classifier predictions for sequential data with Hidden Markov Models (HMMs).',
    long_description=long_description,
    python_requires=">=3.4",
    install_requires=[
        "pandas",
        "joblib",
        "tqdm",
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ]
)
