# PyNN

Simple Python3 Recurrent Neural Networks framework.

## Model
In PyNN structure and state of network are divided. This allows multiple states for single network, that also helpful for parallel learning.

### Structure
The fundamental unit of PyNN network structure is [Node](#Node). 
There are some common operations implementations for Nodes, so-called [Elements](#Element). 
Nodes can be connected with each other and encapsulated in [Network](#Network) which is also Node.

### State
Learned state of network like biases and matrix weights is represented by [_State](#_State). 
Data that need to be saved through forward propagation, e.g. data in loopbacks or Nodes inputs/outputs, is stored in [_Memory](#_Memory). 
Errors for backpropagation are stored in [_Error](#_Error).
Gradient of variables of network, encapsulated in _State, is represented by [_Gradient](#_Gradient).
[_Context](#_Context) contains all these states and also some temporary buffers needed for data propagation.

## `Element`

There are predefined types of element:

+ `MatrixElement`
    + `Matrix`
+ `VectorElement`
    + `Bias`
    + `Tanh`
    + `Rectifier`
+ `Mixer`
    + `Fork`
    + `Join`

You can define your own.

## TODO
- [x] AdaGrad
- [x] Separate structure and state
- [x] Propagation memory
- [ ] Learn initial state
- [ ] Complex numbers support
- [ ] Loss layers
- [ ] Async layers
- [ ] Pre-defined common networks
- [ ] Multithreading
- [ ] GPU Acceleration
