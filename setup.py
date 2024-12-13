#!/usr/bin/env python
"""
OneCircuit: One solution for all quantum circuit needs
"""
import os

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# all information about onecircuit is here
MAN = 0
SUB = 0
SUB1 = 1

VERSION = '%d.%d.%d' % (MAN,SUB, SUB1)
REQUIRES = ['numpy (>=1.8)','scipy (>=1.7.1)']
PACKAGES = ['onecircuit']

NAME = "onecircuit"
AUTHOR = ("Binho Le")
AUTHOR_EMAIL = ("binho@fris.tohoku.ac.jp")
LICENSE = "GNU"
DESCRIPTION = "One solution for all quantum circuit needs."
KEYWORDS = "quantum circuit, variationa quantum circuit"
URL = ""
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]

def write_version_py(filename='onecircuit/_version.py'):
    cnt = """\
#this file is generated from onecircuit setup.py
version = '%(version)s'
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION})
    finally:
        a.close()

# if exists then remove
if os.path.exists('onecircuit/_version.py'):
    os.remove('onecircuit/_version.py')



write_version_py()

setup(name = NAME,
        version = VERSION,
        description=DESCRIPTION,
        url=URL,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        packages=PACKAGES,
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=">=3.6",
)


#find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
