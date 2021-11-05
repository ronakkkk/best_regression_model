import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.4'
PACKAGE_NAME = 'bestregressionmodel'
AUTHOR = 'Ronak Bhagchandani'
AUTHOR_EMAIL = 'rishibhagchandani123@gmail.com'
URL = 'https://github.com/ronakkkk/bestregressionmodel'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'It helps to find the best Regression model with the help of the Root Mean Square Value (RMSE) for the given regression model based on the given dataset'
LONG_DESCRIPTION = ("README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'sklearn'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
