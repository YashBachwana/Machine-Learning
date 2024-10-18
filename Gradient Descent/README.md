# Gradient Descent Algorithms Comparison

This project implements different variants of gradient descent algorithms to demonstrate their convergence behavior and efficiency. The following notebooks are included:

- `fullbatch_gd.ipynb`: Implementation of Full Batch Gradient Descent (GD).
- `momentum_gd.ipynb`: Implementation of Full Batch Gradient Descent with Momentum.
- `momentum_sgd.ipynb`: Implementation of Stochastic Gradient Descent (SGD) with Momentum.
- `stochastic_gd.ipynb`: Implementation of Stochastic Gradient Descent (SGD).

## Comparison of Algorithms

The table below compares the average number of steps taken to converge to an epsilon-neighborhood of the minimizer for two datasets. We compare both vanilla gradient descent and its momentum variants, for both full batch and stochastic cases.

### Data Set Comparison

| Data Set      | GD                   | SGD                           | GD_Momentum                    | SGD_Momentum                   |
| ------------- | -------------------- | ----------------------------- | ------------------------------ | ------------------------------ |
| **Data Set 1**| Diverges              | Epochs: 37, Iterations: 1516   | Epochs: 168                    | Epochs: 16, Iterations: 664     |
| **Data Set 2**| Epochs: 92           | Epochs: 92, Iterations: 3707   | Epochs: 38                     | Epochs: 22, Iterations: 912     |

### Key Insights

1. **Momentum Enhances Stability**:
   - Momentum forces the noisy loss function to approach the optima more precisely compared to standard algorithms.
   - By considering the previous gradient updates, momentum helps avoid overshoot and oscillations, leading to more stable convergence.
   
2. **Accelerated Convergence**:
   - The momentum-based algorithms require fewer epochs to converge, showcasing their ability to accelerate the convergence process. 
   - Fewer epochs and iterations indicate that momentum assists in reaching the optima faster than vanilla gradient descent.
