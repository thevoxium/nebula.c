
#include "include/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline double elapsed_sec(struct timespec start, struct timespec end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(void) {
  printf("MatMul timing (CLOCK_MONOTONIC)\n");
  printf("--------------------------------\n");
  printf("%-13s | %-16s\n", "Matrix Size", "Time");
  printf("--------------------------------\n");

  for (int n = 2; n <= 4096; n *= 2) {
    int shape[2] = {n, n};

    // Create fresh inputs for this size so we don't time allocations.
    Tensor a = random_like(shape, 2, true);
    Tensor b = random_like(shape, 2, true);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    Tensor c = matmul(&a, &b);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double t = elapsed_sec(start, end);
    printf("%4dx%-7d | %10.6f seconds\n", n, n, t);

    free_tensor(&a);
    free_tensor(&b);
    free_tensor(&c);
  }

  printf("--------------------------------\n");
  return 0;
}

// #include "include/tensor.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
//
// int main() {
//   int shape[] = {2, 2};
//
//   Tensor a = make_tensor(shape, 2, (float[]){1, 2, 3, 4}, true);
//   Tensor b = make_tensor(shape, 2, (float[]){1, 2, 3, 4}, true);
//
//   struct timespec start, end;
//   clock_gettime(CLOCK_MONOTONIC, &start);
//
//   Tensor c = matmul(&a, &b);
//
//   clock_gettime(CLOCK_MONOTONIC, &end);
//
//   double elapsed =
//       (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
//
//   printf("Time taken for matmul: %.9f seconds\n", elapsed);
//
//   print_tensor(&a);
//   print_tensor(&b);
//   print_tensor(&c);
//
//   free_tensor(&a);
//   free_tensor(&b);
//   free_tensor(&c);
//
//   return 0;
// }
