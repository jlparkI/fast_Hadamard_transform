from setuptools import setup, Extension
import numpy, os


def main():
    extension_mod = Extension("fastHadamardTransform",
                    [os.path.join("src", "hadamard_transformmodule.c")],
                    include_dirs=[numpy.get_include()])

    setup(name = "fastHadamardTransform", ext_modules=[extension_mod])


if __name__ == "__main__":
    main()
