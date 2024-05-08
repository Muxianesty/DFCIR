# DFCIR (DataFlow Computer Intermediate Representation)
DFCIR is an intermediate representation based on LLVM MLIR project for high-level synthesis of streaming dataflow computers.

It utilizes high-level abstract concepts like `Streams`, `Scalars`, `Kernels` to represent the computational logic of a streaming dataflow computer.
Currently 

Currently DFCIR is supposed to be used in conjunction with FIRRTL intermediate representation from [CIRCT](https://github.com/llvm/circt/) project to generate SystemVerilog-designs for streaming dataflow computers.

The computational logic of a streaming dataflow computer has to be statically scheduling: DFCIR currently has two implementations of scheduling algorithms (Linear Programming-approach and a simple as-soon-as-possible algorithm).

## Prerequisites

**[!!!]** Current version was tested _**only on Ubuntu 20.04**_ **[!!!]**


##
