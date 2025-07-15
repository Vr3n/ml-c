#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Training Dataset.
float train[][2] = {
    {21.225096797004003, 2936.6934049648135},
    {27.628240927476114, 2774.4350053334374},
    {31.782449493664746, 1444.5128725254267},
    {16.874551899214413, 788.0917418045697},
    {16.795350602746595, 249.05657799498843},
    {22.98396559836627, 2785.0624650896298},
    {31.058488837102548, 1907.1364367521733},
    {24.062254648661927, 3000.00},
    {34.02935090601364, 806.9754719684531},
    {19.38997746538984, 871.2646328842654}
};



#define train_count (sizeof(train)/sizeof(train[0]))

// y = x * w + b; -> The Model. (Linear Regression.)
float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX; // type casting: 0 to 1.
}

float mse(float weight, float bias)
{
    // Accumulating the Gradients.
    float result = 0.0f;

    for (size_t i = 0; i < train_count; ++i)
    {
        float x = train[i][0];
        float y = x * weight + bias;
        float d = y - train[i][1];
        result += d*d;
    }
    result /= train_count;

    return result;
}

void autodiff(float w, float b, float *cost_value, float *dw, float *db)
{

    for (size_t i = 0; i < train_count; ++i)
    {
        float x = train[i][0];
        float target = train[i][1];

        // Forward Pass.
        float y = x * w + b;

        float y_dist = y - target;
        *cost_value += y_dist * y_dist;

        // Backward Pass.
        float diff_y = y_dist * 2;
        *dw += diff_y * x;
        *db += diff_y * 1;
    }

    *cost_value /= train_count;
    *dw /= train_count;
    *db /= train_count;

}

int main()
{
    srand(42);
    float cost_value = 0.0f, dw = 0.0f, db = 0.0f;

    // Initializing random variable.
    float w = (rand_float() - 0.05f) * 0.1f;
    float b = (rand_float() - 0.05f) * 0.1f;

    // Epsilon to offset from 0.
    float eps = 1e-3;

    // Learning Rate hyperparameter.
    float alpha = 0.0001;

    const int EPOCHS = 10;

    for (size_t i = 0; i < EPOCHS; ++i) {
        autodiff(w, b, &cost_value, &dw, &db);

        // Applying the Gradient.
        w -= alpha * dw;
        b -= alpha * db;

        printf("%d: w = %f, b = %f, cost = %f\n", i, w, b, cost_value);
    }
    printf("w = %f, b = %f, cost = %f\n", w, b, cost_value);

    return 0;
}
