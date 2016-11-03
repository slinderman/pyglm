from distutils.core import setup

import numpy as np

setup(name='pyglm',
      version='0.1',
      description='Bayesian inference for generalized linear models of neural spike trains',
      author='Scott Linderman',
      author_email='scott.linderman@columbia.edu',
      url='http://www.github.com/slinderman/pyglm',
      include_dirs=[np.get_include(),],
          install_requires=[
              'numpy>=1.9.3', 'scipy>=0.16', 'matplotlib', 'pybasicbayes', 'pypolyagamma'],
      classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++',
      ],
      keywords=[
          'generalized linear model', 'autoregressive', 'AR', 'computational neuroscience'],
      platforms="ALL",
      packages=['pyglm', 'pyglm.utils'])
