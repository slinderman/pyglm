from distutils.core import setup
from Cython.Build import cythonize

import numpy as np

setup(name='pyglm',
      version='0.1',
      description='Bayesian inference for generalized linear models of neural spike trains',
      author='Scott Linderman',
      author_email='slinderman@seas.harvard.edu',
      url='http://www.github.com/slinderman/pyglm',
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
          install_requires=[
        'numpy>=1.9.3', 'scipy>=0.16', 'matplotlib', 'pybasicbayes'],
      classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++',
      ],
      keywords=[
        'generalized linear model', 'autoregressive', 'computational neuroscience'],
      platforms="ALL",
      packages=['pyglm', 'pyglm.utils'])
