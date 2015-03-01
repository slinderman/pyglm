# Downsample the data
import gzip, cPickle

# Downsample 10 bins to 1
ds = 10
N = 27

with gzip.open("train.pkl.gz", "r") as f:
    train = cPickle.load(f)

train_ds = train.reshape((-1,ds,N)).sum(1)
with gzip.open("train.pkl.gz", "w") as f:
    cPickle.dump(train_ds, f, protocol=-1)


with gzip.open("test.pkl.gz", "r") as f:
    test = cPickle.load(f)
test_ds = test.reshape((-1,ds,N)).sum(1)
with gzip.open("test.pkl.gz", "w") as f:
    cPickle.dump(test_ds, f, protocol=-1)