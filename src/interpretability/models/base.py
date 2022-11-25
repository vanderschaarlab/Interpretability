import abc
import torch


class BlackBox(torch.nn.Module):
    @abc.abstractmethod
    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the latent representation for the example x
        :param x: input features
        :return:
        """
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the output for the example x
        :param x: input features
        :return:
        """
        return
