#! /usr/bin/env python

# Public Domain (-) 2011 The Bolt Authors.
# See the Bolt UNLICENSE file for details.

import sys

from setuptools import setup

# ------------------------------------------------------------------------------
# Check Version
# ------------------------------------------------------------------------------

if (not hasattr(sys, 'version_info')) or sys.version_info < (2, 6):
    print("ERROR: Bolt only works on Python 2.6+")
    sys.exit(1)

if sys.version_info > (3,):
    print("ERROR: Bolt doesn't work on Python 3.x yet, only on Python 2.6+")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Run Setup
# ------------------------------------------------------------------------------

setup(
    name="bolt",
    author="tav",
    author_email="tav@espians.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: Public Domain",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Build Tools",
        "Topic :: System :: Clustering",
        "Topic :: System :: Software Distribution",
        "Topic :: System :: Systems Administration"
        ],
    description="Multi-server automation and deployment toolkit",
    entry_points=dict(console_scripts=[
        "bolt = bolt.core:main"
        ]),
    install_requires=[
        'paramiko >=1.7.6',
        'pycrypto >= 1.9',
        "PyYAML>=3.09",
        "tavutil>=1.0"
        ],
    keywords=["admin", "deployment", "ssh"],
    license="Public Domain",
    long_description=open('README.rst').read(),
    packages=["bolt"],
    url="https://github.com/tav/bolt",
    version="0.9",
    zip_safe=True
    )
