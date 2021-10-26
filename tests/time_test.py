import numpy as np, sys
from scipy.linalg import hadamard
from fastHadamardTransform import fastHadamardArray, fastHadamard2dArray
import timeit

def scipy_act(size):
    random_seed = 123
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10,high=10, size=(size))
    hmat = hadamard(marr.shape[0])
    scipy_results = np.matmul(hmat, marr)

def hmart_act(size):
    random_seed = 123
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10,high=10, size=(size))
    fastHadamardArray(marr)


def time_test():
    size = 2048
    print("For an array of size %s, the scipy version takes:"%size)
    print(timeit.timeit(lambda: scipy_act(size), number=50))
    print("While the module version takes:")
    print(timeit.timeit(lambda: hmart_act(size), number=50))

if __name__ == "__main__":
    time_test()
