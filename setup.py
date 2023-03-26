from setuptools import setup

setup(name='modelfitter',
version='0.1',
# description='Testing installation of Package',
# url='#',
author='Zach Davis',
author_email='zachkdavis00@gmail.com',
license='MIT',
packages=['ModelFitter'],
install_requires=[
  "matplotlib>=3.7.1",
  "numpy==1.24.2",
  "easierplotlib>=0.2",
],
zip_safe=False)
