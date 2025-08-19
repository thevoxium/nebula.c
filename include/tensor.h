#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>

typedef struct Tensor Tensor;
typedef void (*Backward_func)(Tensor *);

typedef struct Tensor {
  int *shape;
  int ndim;
  int *strides;
  float *data;

  // grad variables and backward func here;
  bool requires_grad;
  Backward_func _backward;
  float *grad;
  int num_parents;
  Tensor **parents;
} Tensor;

void free_tensor(Tensor *t);

Tensor make_tensor(int *shape, int ndim, float *array, bool requires_grad);
Tensor from_array(int *shape, int ndim, float *array, bool requires_grad);
Tensor random_like(int *shape, int ndim, bool requires_grad);

Tensor add(const Tensor *a, const Tensor *b);
Tensor sub(const Tensor *a, const Tensor *b);

Tensor sigmoid(const Tensor *a);

void zero_grad(Tensor *t);
void backprop(Tensor *out);
void topo_sort(Tensor *root, Tensor ***sorted, int *count, int *capacity,
               bool *visited);

size_t numel(const Tensor *a);
void print_tensor(Tensor *t);

#endif // TENSOR_H
