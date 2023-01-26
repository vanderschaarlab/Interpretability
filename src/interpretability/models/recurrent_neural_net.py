import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from interpretability.models.base import BlackBox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MortalityGRU(BlackBox):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,  # dropout=drop_prob
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.latent_representation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x, h = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


class ArrowHeadGRU(BlackBox):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,  # dropout=drop_prob
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.latent_representation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x, h = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


class ConvNet(BlackBox):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        kernel_size=3,
        output_dim=1,
        drop_prob=0.2,
        activation_func="sigmoid",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.convInput = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding="same")
        self.convHidden1 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, padding="same"
        )
        self.convHidden2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(hidden_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_dim**2, output_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        if activation_func == "sigmoid":
            self.activation_func = nn.Sigmoid()
        elif activation_func == "softmax":
            self.activation_func = nn.Softmax(dim=-1)
        elif not activation_func:
            self.activation_func = None

    def forward(self, x):
        x = self.latent_representation(x)
        x = self.fc1(x)
        if self.activation_func:
            x = self.activation_func(x)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.relu(self.bn1(self.convInput(x)))
        x = self.relu(self.bn2(self.convHidden1(x)))
        x = self.relu(self.bn3(self.convHidden2(x)))
        x = self.flatten(self.pool(x))
        return x


class GRU(BlackBox):
    def __init__(
        self, input_dim=1, hidden_dim=5, output_dim=1, n_layers=3, drop_prob=0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,  # dropout=drop_prob
        )
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.latent_representation(x)
        x = self.fc1(x)
        x = self.sigmoid(
            x,
        )
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x, h = self.gru(x)
        x = x[:, -1, :]
        return x


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def latent_representation(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))

        return hn[0]

    def forward(self, x):
        x = self.latent_representation(x)
        out = self.linear(
            x
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out
