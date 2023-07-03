import numpy as np
import torch
from torch import nn


class LogisticRegression:
    def __init__(
        self,
        features: int,
        max_iter: int,
        l2_strength: float,
        verbose: bool = True,
    ) -> None:
        self.l2_strength = l2_strength
        self.verbose = verbose

        self.model = nn.Linear(features, 1)
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=max_iter)

    def fit(
        self,
        input: np.ndarray,  # [N, C]
        labels: np.ndarray,  # [N]
    ) -> None:
        # Reset the weights
        nn.init.zeros_(self.model.weight)
        nn.init.zeros_(self.model.bias)

        # Make the inputs as tensors
        input_tensor = torch.tensor(input)  # [N, C]
        labels_tensor = torch.tensor(labels, dtype=torch.float,)[
            :, np.newaxis
        ]  # [N, 1]

        # Define the closure
        pos_rate = labels_tensor.mean(dim=0)  # [1]
        pos_weight = (1.0 - pos_rate) / pos_rate  # [1]

        def closure():
            self.optimizer.zero_grad()
            logits = self.model(input_tensor)  # [N, 1]
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels_tensor,
                reduction="sum",
                pos_weight=pos_weight,
            )  # []
            l2_penalty = 0.5 * self.l2_strength * self.model.weight.square().sum()  # []
            loss = bce_loss + l2_penalty
            if self.verbose:
                print(f"BCE loss: {bce_loss.item()}. L2 penalty: {l2_penalty.item()}")

            loss.backward()
            return loss

        # Let's fit
        self.optimizer.step(closure)

    @torch.no_grad()
    def predict_proba(
        self,
        input: np.ndarray,  # [N, C]
    ) -> np.ndarray:  # [N]
        input_tensor = torch.tensor(input)  # [N, C]
        logits = self.model(input_tensor)  # [N, 1]
        proba = torch.sigmoid(logits)[:, 0]  # [N]
        return proba

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
