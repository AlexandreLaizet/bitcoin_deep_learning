from setuptools import find_packages
from setuptools import setup
import platform

if platform.machine() == 'arm64': # M1 machine
    requirements_path = 'requirements_arm64.txt'
else:
    requirements_path = 'requirements.txt'

with open(requirements_path) as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='bitcoin_deep_learning',
      version="1.0",
      description="Project Description",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/bitcoin_deep_learning-run'],
      zip_safe=False)
