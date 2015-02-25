"""
Parse the RGC data into the desired format
This doesn't use the stimulus at all
"""
import numpy as np
from scipy.io import loadmat
import cPickle

raw_data = loadmat("rgc_full.mat", squeeze_me=True)

dt = 0.001

N = raw_data["K"]
T = raw_data["T"] / dt

Ci = raw_data["C"] - 1  # Array of neural indices (0-26)
assert np.amin(Ci) == 0 and np.amax(Ci) == N-1
Si = raw_data["S"]      # Array of spike times (real valued)

# Create the spike train matrix from the array of spike times
S = np.zeros((T,N), dtype=np.int)
for n in xrange(N):
    S[(Si[Ci==n] // dt).astype(np.int), n] = 1

parsed_data = {"S": S, "dt": dt, "N": N, "T": T}

with open("rgc_full.pkl", "w") as f:
    cPickle.dump(parsed_data, f, protocol=-1)

# Save segments of the data for training vs testing
for T_train in [60, 120, 300]:
    end = int(T_train / dt)
    S_train = S[:end, :]
    train_data = {"S": S_train, "dt": dt, "N": N, "T": end}

    with open("rgc_%dT.pkl" % T_train, "w") as f:
        cPickle.dump(train_data, f, protocol=-1)

# Save some data for testing
T_test_start = 301
T_test_stop = 360

start = int(T_test_start / dt)
end = int(T_test_stop / dt)
S_test = S[start:end, :]
test_data = {"S": S_test, "dt": dt, "N": N, "T": end - start}

with open("rgc_test.pkl", "w") as f:
    cPickle.dump(test_data, f, protocol=-1)

