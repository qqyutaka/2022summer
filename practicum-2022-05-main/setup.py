# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

from setuptools import setup, find_packages

# Install for development by running this command in the current folder (the folder containing this
# setup.py file):
#   pip install -e .
# The -e option stands for "editable" and is what tells it to instal in development mode (where it
# installs a link to your current files that are under development instead of copying them).

setup(
    name="audiot",
    version="0.0.0.dev1",
    author="AudioT Inc.",
    author_email="bcarroll@audiot.ai",
    packages=find_packages(),
    # url="",
    # license="LICENSE.txt",
    # description="TODO",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    install_requires=[],
    extras_require={"test": ["pytest"]},
    # entry_points={"console_scripts":["script_name=module.path:func_name"]},
)
