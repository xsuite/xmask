# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from setuptools import setup, find_packages, Extension
from pathlib import Path

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []

#########
# Setup #
#########

version_file = Path(__file__).parent / 'xmask/_version.py'
dd = {}
with open(version_file.absolute(), 'r') as fp:
    exec(fp.read(), dd)
__version__ = dd['__version__']

setup(
    name='xmask',
    version=__version__,
    description='Configuration of tracking simulations for the LHC and other accelerators',
    long_description='Configuration of tracking simulations for the LHC and other accelerator',
    url='https://github.com/xsuite/xsuite/issues',
    author='G. Iadarola et al.',
    license='Apache 2.0',
    download_url="https://pypi.python.org/pypi/xmask",
    project_urls={
            "Bug Tracker": "https://github.com/xsuite/xsuite/issues",
            "Source Code": "https://github.com/xsuite/xmask/",
        },
    packages=find_packages(),
    ext_modules = extensions,
    include_package_data=True,
    install_requires=[
        'numpy>=1.0',
        'pyyaml',
        'xfields',
        ],
    extras_require={
        'tests': ['cpymad', 'pytest'],
        },
    )
