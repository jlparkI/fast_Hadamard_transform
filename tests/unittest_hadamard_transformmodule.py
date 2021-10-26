import unittest, numpy as np, time
from scipy.linalg import hadamard
from fastHadamardTransform import fastHadamardArray, fastHadamard2dArray

#To test the C extension for the fast hadamard transform, we compare the 
#results to matrix multiplication with Hadamard matrices generated by
#Scipy (result should be the same).
class TestFastHadamardTransform(unittest.TestCase):

    def test_array_transform(self):
        random_seed = 123
        rng = np.random.default_rng(random_seed)
        marr = rng.uniform(low=-10,high=10, size=(256))
        hmat = hadamard(marr.shape[0])
        scipy_results = np.matmul(hmat, marr)
        corrected_arr = np.copy(marr)
        fastHadamardArray(corrected_arr)
        outcome = np.allclose(corrected_arr, scipy_results)
        print("Did the C extension provide the correct result for "
                    "a %s array? %s"%(hmat.shape[0], outcome))
        self.assertTrue(outcome)

    def test_2d_array_transform(self):
        random_seed = 123
        rng = np.random.default_rng(random_seed)
        marr = rng.uniform(low=-10.0,high=10.0, size=(100,256))
        
        hmat = hadamard(marr.shape[1])
        scipy_results = np.matmul(hmat, marr.T).T
        fastHadamard2dArray(marr)
        outcome = np.allclose(marr, scipy_results)
        print("**********\nDid the C extension provide the correct result for columnwise "
            "transforms of %s %s-column 2d arrays? %s\n*******"%(marr.shape[0], marr.shape[1], outcome))
        self.assertTrue(outcome)



if __name__ == "__main__":
    unittest.main()