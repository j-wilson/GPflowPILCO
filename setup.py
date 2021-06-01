from setuptools import setup, find_packages

requirements = (
  'gpflow>=2.2.1',
  'gpflow-sampling>=0.2',
  'gym>=0.17.3',
  'numpy',
  'scipy',
  'sklearn',
  'tf-nightly',
  'tfp-nightly',
  'tqdm',
)

setup(
    name='gpflow_pilco',
    version='0.1',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=requirements
)