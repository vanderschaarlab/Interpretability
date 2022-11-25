import torch
import torch.nn as nn
import torch.nn.functional as F
from interpretability.models.base import BlackBox


class DiabetesMLPRegressor(BlackBox):
    def __init__(self, input_feature_num=10) -> None:
        """
        Mortality predictor MLP
        :param n_cont: number of continuous features among the output features
        """
        super().__init__()
        self.lin1 = nn.Linear(input_feature_num, 200)
        self.lin2 = nn.Linear(200, 50)
        self.lin3 = nn.Linear(50, 1)
        self.drops = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_representation(x)
        x = self.lin3(x)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).detach().numpy()


class IrisMLP(BlackBox):
    def __init__(self, n_cont: int = 3, input_feature_num=26) -> None:
        """
        Mortality predictor MLP
        :param n_cont: number of continuous features among the output features
        """
        super().__init__()
        self.n_cont = n_cont
        self.lin1 = nn.Linear(input_feature_num, 200)
        self.lin2 = nn.Linear(200, 50)
        self.lin3 = nn.Linear(50, 3)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.drops = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_cont, x_disc = x[:, : self.n_cont], x[:, self.n_cont :]
        x_cont = self.bn1(x_cont)
        x = torch.cat([x_cont, x_disc], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        return x

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the class probabilities for the input x
        :param x: input features
        :return: probabilities
        """
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.softmax(x, dim=-1)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.probabilities(x)
        preds = torch.argmax(probs)
        return preds

    def latent_to_presoftmax(self, h: torch.Tensor) -> torch.Tensor:
        """
        Maps a latent representation to a preactivation output
        :param h: latent representations
        :return: presoftmax activations
        """
        h = self.lin3(h)
        return h


class WineMLP(BlackBox):
    def __init__(self, n_cont: int = 11, input_feature_num=11) -> None:
        """
        Mortality predictor MLP
        :param n_cont: number of continuous features among the output features
        """
        super().__init__()
        self.n_cont = n_cont
        self.lin1 = nn.Linear(input_feature_num, 200)
        self.lin2 = nn.Linear(200, 50)
        self.lin3 = nn.Linear(50, 7)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.drops = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_cont, x_disc = x[:, : self.n_cont], x[:, self.n_cont :]
        x_cont = self.bn1(x_cont)
        x = torch.cat([x_cont, x_disc], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        return x

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the class probabilities for the input x
        :param x: input features
        :return: probabilities
        """
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.softmax(x, dim=-1)
        return x

    def latent_to_presoftmax(self, h: torch.Tensor) -> torch.Tensor:
        """
        Maps a latent representation to a preactivation output
        :param h: latent representations
        :return: presoftmax activations
        """
        h = self.lin3(h)
        return h
