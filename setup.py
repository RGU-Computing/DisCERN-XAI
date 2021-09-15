import setuptools

VERSION_STR = "0.0.1"

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
     name='discern-xai',
     version=VERSION_STR,
     author="Anjana Wijekoon",
     author_email="a.wijekoon1@rgu.ac.uk",
     description="DisCERN: Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/RGU-Computing/discern-xai",
     packages=setuptools.find_packages(exclude=("test",)),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
     ],
     keywords='machine-learning explanation interpretability counterfactual',
     install_requires=install_requires,
 )
