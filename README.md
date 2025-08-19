# nebula.c

**nebula.c** is a minimalistic C library for tensor operations and automatic differentiation



## Getting Started

### Prerequisites

-   **C99 or newer** compiler.
    
-   For accelerated math on macOS:  
    Define `USE_ACCELERATE` during compilation.
    

### Building

To build a project using **nebula.c**, simply include `tensor.c` and `tensor.h` in your source files:

For Apple Accelerate support (macOS only):

```bash
clang -Ofast -flto -ffast-math -march=native -DUSE_ACCELERATE -o tensor main.c src/*.c -framework Accelerate && ./tensor
```

If you do not have accelerate framework, you can remove the -DUSE_ACCELERATE and -framework Accelerate flags


### Basic Usage

```c
#include "include/tensor.h"

int shape[2] = {2, 2};
float data[4] = {1.0, 2.0, 3.0, 4.0};
Tensor a = from_array(shape, 2, data, true);
Tensor b = random_like(shape, 2, true);

Tensor c = add(&a, &b);
Tensor d = sigmoid(&c);

backprop(&d); // Compute gradients

print_tensor(&d);
free_tensor(&a);
free_tensor(&b);
free_tensor(&c);
free_tensor(&d);

```

## Memory Management

All tensors allocated via the API should be freed with `free_tensor` to avoid memory leaks.
