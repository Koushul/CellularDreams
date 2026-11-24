#include <stddef.h>

/* R's cumsum accumulates in long double, then rounds each partial sum to double. */
void r_cumsum(const double *x, double *y, size_t n) {
    long double sum = 0.0L;
    for (size_t i = 0; i < n; i++) {
        sum += (long double)x[i];
        y[i] = (double)sum;
    }
}

/* R's mean() sums in long double. */
double r_mean(const double *x, size_t n) {
    long double sum = 0.0L;
    for (size_t i = 0; i < n; i++) {
        sum += (long double)x[i];
    }
    return (double)(sum / (long double)n);
}
