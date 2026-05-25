import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)

        current_weights = initial_weights
        for i in range(num_iterations):
            # Getting predictions:
            y_preds = self.get_model_prediction(X, current_weights)

            # Getting gradients and updating weights:
            for j in range(len(initial_weights)):
                weight_grad = self.get_derivative(
                    model_prediction=y_preds,
                    ground_truth=Y,
                    N=len(X),
                    X=X,
                    desired_weight=j
                )
                current_weights[j] -= self.learning_rate * weight_grad

        return np.round(current_weights, 5)







