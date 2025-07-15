#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Training Dataset.(OR-gate)
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
};

#define train_count (sizeof(train)/sizeof(train[0]))

// y = x * w + b; -> The Model. (Linear Regression.)
float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX; // type casting: 0 to 1.
}

float mse(float w1, float w2, float b)
{
    // Accumulating the Gradients.
    float result = 0.0f;

    for (size_t i = 0; i < train_count; ++i)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = x1 * w1 + x2 * w2 + b;
        float d = y - train[i][2];
        result += d*d;
    }
    result /= train_count;

    return result;
}

int main (void)
{
    srand(42);

    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float eps = 1e-3;
    float alpha = 0.01;

    printf("w1 = %f, w2 = %f\n", w1, w2);

    for (size_t i = 0; i < 100; ++i) {
        float cost = mse(w1, w2, b);
        float dw1 = (mse(w1 + eps, w2, b) - cost) / eps;
        float dw2 = (mse(w1, w2 + eps, b) - cost) / eps;
        float db = (mse(w1, w2, b + eps) - cost) / eps;

        w1 -= dw1 * alpha;
        w2 -= dw2 * alpha;
        b -= db * alpha;
        printf("Epoch %d: Cost: %f, w1: %f, w2: %f, b: %f\n",i, cost, w1, w2, b);
    }
}
