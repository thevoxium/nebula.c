#include "include/tensor.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  int shape[] = {2, 2};
  float data_a[] = {1.0, 2.0, 3.0, 4.0};
  float data_b[] = {0.5, 1.5, 2.5, 3.5};

  Tensor a = make_tensor(shape, 2, data_a, true); // requires_grad=true
  Tensor b = make_tensor(shape, 2, data_b, true);

  printf("Tensor a:\n");
  print_tensor(&a);
  printf("Tensor b:\n");
  print_tensor(&b);

  Tensor sum = add(&a, &b);
  Tensor c = sigmoid(&sum);

  print_tensor(&c);

  backprop(&c);

  printf("Gradients of a:\n");
  for (size_t i = 0; i < numel(&a); i++) {
    printf("%f ", a.grad[i]);
  }
  printf("\n");

  printf("Gradients of b:\n");
  for (size_t i = 0; i < numel(&b); i++) {
    printf("%f ", b.grad[i]);
  }
  printf("\n");

  free_tensor(&a);
  free_tensor(&b);
  free_tensor(&sum);
  free_tensor(&c);

  return 0;
}
