import numpy as np


class Loss:
    def calculate(self, outputs, expected):
        sample_losses = self.forward(outputs, expected)
        return np.mean(sample_losses)


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clipping is to avoid getting a division by zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        return -np.log(correct_confidences)
