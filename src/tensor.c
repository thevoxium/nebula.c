#include "../include/tensor.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__) && defined(USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define HAVE_ACCELERATE 1
#endif

bool check_shape(const Tensor *a, const Tensor *b) {
  if (a->ndim != b->ndim) {
    return 0;
  } else {
    for (size_t i = 0; i < a->ndim; i++) {
      if (a->shape[i] != b->shape[i]) {
        return 0;
      }
    }
  }
  return 1;
}

size_t numel(const Tensor *a) {
  size_t num_elements = 1;
  for (size_t i = 0; i < a->ndim; i++) {
    num_elements *= a->shape[i];
  }
  return num_elements;
}

Tensor make_tensor(int *shape, int ndim, float *array, bool requires_grad) {
  Tensor t;
  t.ndim = ndim;
  t.shape = malloc(ndim * sizeof(int));
  memcpy(t.shape, shape, ndim * sizeof(int));

  t.strides = malloc(ndim * sizeof(int));
  size_t stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    t.strides[i] = stride;
    stride *= shape[i];
  }

  size_t num_elements = numel(&t);
  t.data = malloc(num_elements * sizeof(float));
  if (array != NULL) {
    for (size_t i = 0; i < num_elements; ++i) {
      t.data[i] = array[i];
    }
  }

  if (requires_grad) {
    t.grad = calloc(num_elements, sizeof(float));
  } else {
    t.grad = NULL;
  }
  t.requires_grad = requires_grad;
  t._backward = NULL;
  t.num_parents = 0;
  t.parents = NULL;

  return t;
}

Tensor random_like(int *shape, int ndim, bool requires_grad) {
  Tensor t = make_tensor(shape, ndim, NULL, requires_grad);
  size_t num_elements = numel(&t);
  for (size_t i = 0; i < num_elements; ++i) {
    t.data[i] = (float)rand() / RAND_MAX;
  }
  return t;
}

Tensor from_array(int *shape, int ndim, float *array, bool requires_grad) {
  Tensor t = make_tensor(shape, ndim, array, requires_grad);
  return t;
}

void add_backward(Tensor *out) {
  Tensor *a = out->parents[0];
  Tensor *b = out->parents[1];

  size_t num_elements = numel(out);

  if (a->requires_grad) {
    for (size_t i = 0; i < num_elements; ++i) {
      a->grad[i] += out->grad[i];
    }
  }
  if (b->requires_grad) {
    for (size_t i = 0; i < num_elements; ++i) {
      b->grad[i] += out->grad[i];
    }
  }
}

void sub_backward(Tensor *out) {
  Tensor *a = out->parents[0];
  Tensor *b = out->parents[1];

  size_t num_elements = numel(out);

  if (a->requires_grad) {
    for (size_t i = 0; i < num_elements; ++i) {
      a->grad[i] += out->grad[i];
    }
  }
  if (b->requires_grad) {
    for (size_t i = 0; i < num_elements; ++i) {
      b->grad[i] -= out->grad[i];
    }
  }
}

void sigmoid_backward(Tensor *out) {
  Tensor *a = out->parents[0];
  size_t num_elements = numel(out);

  if (a->requires_grad) {
    for (size_t i = 0; i < num_elements; ++i) {
      a->grad[i] += (out->grad[i] * out->data[i] * (1.0f - out->data[i]));
    }
  }
}

Tensor add(const Tensor *a, const Tensor *b) {
  if (!check_shape(a, b)) {
    fprintf(stderr, "add_tensors error: shape mismatch\n");
    return (Tensor){0};
  }
  Tensor out = make_tensor(a->shape, a->ndim, NULL,
                           a->requires_grad || b->requires_grad);
  size_t num_elements = numel(a);
  for (size_t i = 0; i < num_elements; i++) {
    out.data[i] = a->data[i] + b->data[i];
  }
  if (out.requires_grad) {
    out.parents = malloc(2 * sizeof(Tensor *));
    out.num_parents = 2;
    out.parents[0] = (Tensor *)a;
    out.parents[1] = (Tensor *)b;
    out._backward = add_backward;
  }
  return out;
}

Tensor sub(const Tensor *a, const Tensor *b) {
  if (!check_shape(a, b)) {
    fprintf(stderr, "add_tensors error: shape mismatch\n");
    return (Tensor){0};
  }
  Tensor out = make_tensor(a->shape, a->ndim, NULL,
                           a->requires_grad || b->requires_grad);
  size_t num_elements = numel(a);
  for (size_t i = 0; i < num_elements; i++) {
    out.data[i] = a->data[i] - b->data[i];
  }
  if (out.requires_grad) {
    out.parents = malloc(2 * sizeof(Tensor *));
    out.num_parents = 2;
    out.parents[0] = (Tensor *)a;
    out.parents[1] = (Tensor *)b;
    out._backward = sub_backward;
  }
  return out;
}

Tensor sigmoid(const Tensor *a) {
  Tensor out = make_tensor(a->shape, a->ndim, NULL, a->requires_grad);

  const size_t num_elements = numel(a);

#ifdef HAVE_ACCELERATE
  float *temp = malloc(num_elements * sizeof(float));
  float half = 0.5f, one = 1.0f;
  vDSP_vsmul(a->data, 1, &half, temp, 1, num_elements);
  vvtanhf(temp, temp, (int *)&num_elements);
  vDSP_vsmsa(temp, 1, &half, &half, out.data, 1, num_elements);
  free(temp);
#else
  for (size_t i = 0; i < num_elements; i++) {
    out.data[i] = 1.0 / (1 + expf(-a->data[i]));
  }
#endif

  if (a->requires_grad) {
    out.num_parents = 1;
    out.parents = malloc(sizeof(Tensor *));
    out.parents[0] = (Tensor *)a;
    out._backward = sigmoid_backward;
  }

  return out;
}

void topo_sort(Tensor *root, Tensor ***sorted, int *count, int *capacity,
               bool *visited) {
  uintptr_t addr = (uintptr_t)root;
  size_t hash = addr % 10007;
  if (visited[hash]) {
    return;
  } else {
    visited[hash] = true;
  }

  for (int i = 0; i < root->num_parents; ++i) {
    topo_sort(root->parents[i], sorted, count, capacity, visited);
  }

  if (*count > *capacity) {
    *capacity *= 2;
    *sorted = realloc(*sorted, *capacity * sizeof(Tensor *));
  }
  (*sorted)[(*count)++] = root;
}

void backprop(Tensor *root) {
  if (root->requires_grad == false) {
    fprintf(stderr, "Root does not require backprop calculations");
    return;
  }

  size_t num_elements = numel(root);
  for (size_t i = 0; i < num_elements; ++i) {
    root->grad[i] = 1.0f;
  }
  int capacity = 100;
  int count = 0;
  Tensor **sorted = malloc(capacity * sizeof(Tensor *));
  bool *visited = calloc(10007, sizeof(bool));

  topo_sort(root, &sorted, &count, &capacity, visited);

  for (int i = count - 1; i >= 0; --i) {
    if (sorted[i]->_backward) {
      sorted[i]->_backward(sorted[i]);
    }
  }
  free(sorted);
  free(visited);
}

void zero_grad(Tensor *t) {
  if (t->grad) {
    size_t num_elements = numel(t);
    for (size_t i = 0; i < num_elements; ++i) {
      t->grad[i] = 0.0f;
    }
  }
}

void free_tensor(Tensor *t) {
  free(t->data);
  free(t->shape);
  free(t->strides);
  free(t->grad);
  free(t->parents);

  t->data = NULL;
  t->shape = NULL;
  t->strides = NULL;
  t->grad = NULL;
  t->parents = NULL;
  t->ndim = 0;
  t->num_parents = 0;
  t->requires_grad = false;
  t->_backward = NULL;
}

void print_recursive(Tensor *t, int dim, int offset) {
  if (dim == t->ndim - 1) {
    printf("[");
    for (int i = 0; i < t->shape[dim]; i++) {
      int index = offset + i * t->strides[dim];
      printf("%g", t->data[index]);
      if (i < t->shape[dim] - 1)
        printf(", ");
    }
    printf("]");
  } else {
    printf("[");
    for (int i = 0; i < t->shape[dim]; i++) {
      print_recursive(t, dim + 1, offset + i * t->strides[dim]);
      if (i < t->shape[dim] - 1)
        printf(",\n");
    }
    printf("]");
  }
}

void print_tensor(Tensor *t) {
  print_recursive(t, 0, 0);
  printf("\n");
}
