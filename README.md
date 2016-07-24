First, you need to install a few dependencies:
1. You'll need PyPolyaGamma to be in your Python path. You can get it here:
`git clone --recursive git@github.com:slinderman/pypolyagamma.git`
2. If you want to use the switching models, you'll need Matt Johnson's PyHSMM package on your Python path too:
`git clone --recursive git@github.com:mattjj/pyhsmm.git`
Both of these repos have Cython extensions that you will need to compile. See the repo READMEs for more instructions.

Once you've installed the dependencies, clone PyGLM by running:
`git clone --recursive git@github.com:slinderman/pyglm.git`
Then build the Cython extensions with `cd pyglm` and `python setup.py build_ext --inplace`


To generate some synthetic data, run `ipython examples/generate_synthetic_data.py`

The examples directory has demos of fitting data, given as a `TxN` numpy array of spike counts, 
where `T` is the number of time bins and `N` is the number of neurons, using a simple generalized 
linear model with network interactions.

