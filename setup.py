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
      packages=['pyglm', 'pyglm.internals', 'pyglm.utils'])
