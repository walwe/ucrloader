from setuptools import setup, find_packages

__version__ = "0.4"


setup(name='ucrloader',
      version=__version__,
      description='Loader for the UCR time series dataset into numpy arrays'
                  'https://www.cs.ucr.edu/~eamonn/time_series_data/',
      url='https://github.com/walwe/ucrloader',
      packages=find_packages(),
      install_requires=["pandas"]
      )
