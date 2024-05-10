# DFCIR (DataFlow Computer Intermediate Representation)
DFCIR is an intermediate representation based on LLVM MLIR project for high-level synthesis of streaming dataflow computers.

It utilizes high-level abstract concepts like `Streams`, `Scalars`, `Kernels` to represent the computational logic of a streaming dataflow computer.
Currently 

Currently DFCIR is supposed to be used in conjunction with FIRRTL intermediate representation from [CIRCT](https://github.com/llvm/circt/) project to generate SystemVerilog-designs for streaming dataflow computers.

The computational logic of a streaming dataflow computer has to be statically scheduling: DFCIR currently has two implementations of scheduling algorithms (Linear Programming-approach and a simple as-soon-as-possible algorithm).

## Prerequisites

**[!!!]** Current version was tested _**only on Ubuntu 20.04**_ **[!!!]**

In order to build DFCIR a number of dependencies has to be installed:
* `build-essential`
* `clang` + `lld` or `g++` + `gcc` as a C/C++ compiler
* `liblpsolve55-dev`
* `ninja-build` (preferred) or `make`

This can be done using the following command:
```
sudo apt install build-essential clang lld g++ gcc liblpsolve55-dev ninja-build make
```
## Prerequisites (CIRCT)

Additionally, one of the CIRCT libraries' releases is required ([releases](https://github.com/llvm/circt/releases)).

**[!!!]Currently supported release: 1.72.0, Apr 5 2024 ([link](https://github.com/llvm/circt/releases/tag/firtool-1.72.0))[!!!]**

**Note that <ins>it's the libraries that are required</ins>, not the binary executables.**<br>
Look for the archives which have the following names:
`circt-full-static-<ARCH>` or `circt-full-shared-<ARCH>` for static and dynamic libraries respectively.

**[!!!]Current releases have an inconsistency in their configuration files[!!!]:**<br>
after downloading the chosen release archive, extract the files inside and look for the file `lib/cmake/mlir/MLIRTargets.mlir`.

Open `MLIRTargets.mlir` with a text editor of your choice and look the `_NOT_FOUND_MESSAGE_targets` near the end of the file:<br>
for 1.72.0, it's line **3012** for `circt-full-static-linux-x64`-archive and line **3008** for `circt-full-shared-linux-x64.tar`-archive.

Remove all the `"CIRCT*"`-entries from the corresponding `foreach`-statement and save the file.

## Building DFCIR
DFCIR is a CMake package, so in order to build it a simple CMake environment has to be set up.

DFCIR needs to know the location which the chosen release archive was extracted to and **the full path** to this location has to be passed to CMake environment in the form of `CIRCT_BIN_HOME` CMake variable:<br>

assuming CIRCT-archive was downloaded in home directory (`~`) and was extracted using `tar -xvf`, thus `firtool-1.72.0` (the directory with the top-level `CMakeLists.txt`-file) is also in the home directory, the CMake environment can be set up using the following commands:
```
cd <DFCIR_DIR>
mkdir build
cd build
cmake .. -G Ninja -DCIRCT_BIN_HOME="~/firtool-1.72.0"
```
After this, use `cmake --build .` to build DFCIR.
