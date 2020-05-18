"""Twitter2SQL: A toolset for uploading Twitter JSON data pulled from
the Twitter streaming API into a PostgreSQL databases. Not maintained for 
public use. Capabilities may expand in future.
"""

DOCLINES = __doc__.split("\n")

import sys

from setuptools import setup, find_packages
# from codecs import open
# from os import path
# import os

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version 3.5 or greater required.")

setup(
    name='twitter2sql',
    version='0.1.1',
    description=DOCLINES[0],
    packages=find_packages(),
    entry_points={
        "console_scripts": [], 
    },
    author='Andrew Beers',
    author_email='albeers@uw.edu',
    url='https://github.com/AndrewBeers/Twitter2SQL',  # use the URL to the github repo
    download_url='https://github.com/AndrewBeers/Twitter2SQL/0.1.1',
    keywords=['twitter', 'sql'],
    install_requires=[],
    classifiers=[],
)