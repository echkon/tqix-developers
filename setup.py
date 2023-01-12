#!/usr/bin/env python
"""
tqix: a Toolbox for Quantum in X:
   x: quantum measurement, quantum metrology, quantum tomography, and more.
"""
import os

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# all information about tqix is here
MAN = 2
SUB = 0
SUBS= 4
VERSION = '%d.%d.%d' % (MAN,SUB,SUBS)
REQUIRES = ['numpy (>=1.8)', 'torch (>=1.10)','scipy (>=1.7.1)','sklearn']
PACKAGES = ['tqix', 'tqix/dsm', 'tqix/pis', 'tqix/povm']

NAME = "tqix"
AUTHOR = ("Binho Le")
AUTHOR_EMAIL = ("binho@kindai.ac.jp")
LICENSE = "GNU"
DESCRIPTION = "A Toolbox for Quantum in X"
KEYWORDS = "quantum measurement, quantum metrology, quantum tomography"
URL = "https://vqisinfo.wixsite.com/tqix"
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]

def write_version_py(filename='tqix/version.py'):
    cnt = """\
#this file is generated from tqix setup.py
version = '%(version)s'
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION})
    finally:
        a.close()

# if exists then remove
if os.path.exists('tqix/version.py'):
    os.remove('tqix/version.py')



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