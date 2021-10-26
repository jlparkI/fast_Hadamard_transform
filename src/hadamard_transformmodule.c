#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0

static PyObject *fastHadamardArray_(PyObject *self, PyObject *args);
static PyObject *fastHadamard2dArray_(PyObject *self, PyObject *args);
const char *checkArrayInputs(PyArrayObject *inputArray, int *errorCode);
const char *checkNdArrayInputs(PyArrayObject *inputArray, int expectedDims,
		int *errorCode);


//Boilerplate to help Python's setuptools compile this properly.
static PyMethodDef moduleMethods[] = {
    { "fastHadamardArray",fastHadamardArray_,
      METH_VARARGS,
      "Perform an in-place fast Hadamard transform on an input array"},
    { "fastHadamard2dArray",fastHadamard2dArray_,
      METH_VARARGS,
      "Perform an in-place fast Hadamard transform on a stack of input arrays"},
    {NULL, NULL, 0, NULL}
};


//More boilerplate.
PyMODINIT_FUNC PyInit_fastHadamardTransform(void)
{
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "fastHadamardTransform",
        "Perform a fast Hadamard transform on input arrays",
        -1,
        moduleMethods,
        NULL, NULL, NULL, NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    import_array();
    return module;
}


//Performs an unnormalized fast in-place Walsh-Hadamard transform on a 1d numpy array.
static PyObject *fastHadamardArray_(PyObject *self, PyObject *args)
{
    PyArrayObject *inputArray;
    int errorCode;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &inputArray)){
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to "
                "fastHadamardArray.");
        return NULL;
    }
    
    const char *errorMessage = checkArrayInputs(inputArray, &errorCode);
    
    if (errorCode != VALID_INPUTS){
        PyErr_SetString(PyExc_ValueError, errorMessage);
        return NULL;
    }
    int i = 0, j, h = 1;
    double x, y;
    double *currentElement = NULL;

    while (h < inputArray->dimensions[0]){
        i = 0;
	while (i < inputArray->dimensions[0]){
            for (j=i; j < i+h; j++){
                x = *((double *)PyArray_GETPTR1(inputArray, j));
                y = *((double *)PyArray_GETPTR1(inputArray, j+h));
                currentElement = (double *)PyArray_GETPTR1(inputArray, j);
                *currentElement = x + y;
                currentElement = (double *)PyArray_GETPTR1(inputArray, j + h);
                *currentElement = x - y;
            }
            i = i + h * 2;
        }
        h = h * 2;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

//Performs an unnormalized fast hadamard transform on a 2d numpy array. In this case, the
//transform is performed treating each row of the input as a separate 
//1d array and performing a transform on it.
static PyObject *fastHadamard2dArray_(PyObject *self, PyObject *args)
{
    PyArrayObject *inputArray;
    int errorCode;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &inputArray)){
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to "
                "fastHadamard2dArray.");
        return NULL;
    }
    
    int expectedDims = 2;
    const char *errorMessage = checkNdArrayInputs(inputArray, expectedDims,
		    &errorCode);
        
    if (errorCode != VALID_INPUTS){
        PyErr_SetString(PyExc_ValueError, errorMessage);
        return NULL;
    }
    int i = 0, j, h = 1, rownum;
    double x, y;
    double *currentElement = NULL;
    for (rownum=0; rownum < inputArray->dimensions[0]; rownum++){
	h = 1;
        while (h < inputArray->dimensions[1]){
            i = 0;
	    while (i < inputArray->dimensions[1]){
                for (j=i; j < i+h; j++){
                    x = *((double *)PyArray_GETPTR2(inputArray, rownum, j));
                    y = *((double *)PyArray_GETPTR2(inputArray, rownum, j+h));
                    currentElement = (double *)PyArray_GETPTR2(inputArray, rownum, j);
                    *currentElement = x + y;
                    currentElement = (double *)PyArray_GETPTR2(inputArray, rownum, j + h);
                    *currentElement = x - y;
                }
                i = i + h * 2;
            }
            h = h * 2;
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}



//This function checks that the inputs are valid for 1d arrays.
//Note that for these and all other input types, we require that the input
//size must be a power of 2.
const char *checkArrayInputs(PyArrayObject *inputArray, int *errorCode)
{
    double log2Result;
    if (inputArray->nd != 1 || PyArray_TYPE(inputArray) != NPY_DOUBLE){
        *errorCode = INVALID_INPUTS;
        return "input array must be 1d and of type np.float64.";
    }
    if (!(PyArray_FLAGS(inputArray) & NPY_ARRAY_C_CONTIGUOUS)){
        *errorCode = INVALID_INPUTS;
        return "input array must be C-contiguous."; 
    }
    if (inputArray->dimensions[0] < 1){
        *errorCode = INVALID_INPUTS;
        return "the input array must contain at least one element.";
    }
    
    log2Result = log2( (double)inputArray->dimensions[0] );
    if (fabs(round(log2Result) - log2Result) > EPS_TOLERANCE){
        *errorCode = INVALID_INPUTS;
        return "the size of the input array must be a power of 2."; 
    }
    *errorCode = VALID_INPUTS;
    return "all clear."; 
}


//This function checks that the inputs are valid for multidimensional arrays.
const char *checkNdArrayInputs(PyArrayObject *inputArray, int expectedDims,
		int *errorCode)
{
    double log2Result;
    if (inputArray->nd != expectedDims || PyArray_TYPE(inputArray) != NPY_DOUBLE){
        *errorCode = INVALID_INPUTS;
        return "input array must have correct dimensionality and of type np.float64.";
    }
    if (!(PyArray_FLAGS(inputArray) & NPY_ARRAY_C_CONTIGUOUS)){
        *errorCode = INVALID_INPUTS;
        return "input array must be C-contiguous."; 
    }
    if (inputArray->dimensions[1] < 1){
        *errorCode = INVALID_INPUTS;
        return "the input array must contain at least one element.";
    }
    
    log2Result = log2( (double)inputArray->dimensions[1] );
    if (fabs(round(log2Result) - log2Result) > EPS_TOLERANCE){
        *errorCode = INVALID_INPUTS;
        return "the number of columns of the input array must be a power of 2."; 
    }
    *errorCode = VALID_INPUTS;
    return "all clear."; 
}


