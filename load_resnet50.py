'''
DINO architecture taken from https://github.com/lightly-ai/lightly/
'''
import copy
import os
from typing import Optional, Sequence, Union

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.models as models

from resnet50_utils import (
    deactivate_requires_grad, _no_grad_trunc_normal_)


class Backbone(nn.Module):
    '''
    Loading class for the backbone model.
    Backbones are identified by a string name, and the number of input channels.
    '''
    def __init__(
        self,
        name_model: str,
        in_channels: int = 3,
    ):
        super().__init__()
        if in_channels not in {1, 3}:
            raise ValueError("Invalid number of input channels. Should be 1 or 3.")

        self.type_model = 'resnet'
        self.backbone_dict = {
            "resnet18": models.resnet18(weights=None, num_classes=1),
            "resnet50": models.resnet50(weights=None, num_classes=1),
        }
        self.backbone = self.backbone_dict[name_model]

        self.in_features = self.backbone.fc.in_features

        # Modify the first convolutional layer to accept a single channel input
        if in_channels == 1:
            self._modify_first_conv_layer()

        # Removes or replaces the last layer(s)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

    def _modify_first_conv_layer(self):
        """
        In the case of Radar dataset and other single channel images, we need to modify
        the first conv layer to accept 1 input channel instead of 3.
        resnets and regnets have slighlty different naming.
        """
        # Get the first conv layer
        conv1 = self.backbone.conv1

        # Create a new conv layer with 1 input channel instead of 3
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=(conv1.bias is not None))

        with torch.no_grad():
            new_conv1.weight = nn.Parameter(conv1.weight.sum(dim=1, keepdim=True))

        # Replace the original conv1 with the new_conv1
        self.backbone.conv1 = new_conv1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ProjectionHead(nn.Module):
    """
    Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer, use_bias (optional)).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(
        self,
        blocks: Sequence[
            Union[
                tuple[int, int, Optional[nn.Module], Optional[nn.Module]],
                tuple[int, int, Optional[nn.Module], Optional[nn.Module], bool],
            ],
        ]
    ):
        super().__init__()

        layers: list[nn.Module] = []
        for block in blocks:
            input_dim, output_dim, batch_norm, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes one forward pass through the projection head.

        Args:
            x: Input of shape bsz x num_ftrs.
        """
        projection = self.layers(x)
        return projection


class DINOProjectionHead(ProjectionHead):
    """
    Projection head used in DINO.

    "The projection head consists of a 3-layer multi-layer perceptron (MLP)
    with hidden dimension 2048 followed by l2 normalization and a weight
    normalized fully connected layer with K dimensions, which is similar to the
    design from SwAV [1]." [0]

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: SwAV, 2020, https://arxiv.org/abs/2006.09882

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        output_dim:
            The output dimension of the head.
        batch_norm:
            Whether to use batch norm or not. Should be set to False when using
            a vision transformer backbone.
        freeze_last_layer:
            Number of epochs during which we keep the output layer fixed.
            Typically doing so during the first epoch helps training. Try
            increasing this value if the loss does not decrease.
        norm_last_layer:
            Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the
            training unstable.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        batch_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):
        bn = nn.BatchNorm1d(hidden_dim) if batch_norm else None

        super().__init__(
            [
                (input_dim, hidden_dim, bn, nn.GELU()),
                (hidden_dim, hidden_dim, bn, nn.GELU()),
                (hidden_dim, bottleneck_dim, None, None),
            ]
        )
        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        # Tell mypy this is ok because fill_ is overloaded.
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        # Option to normalize last layer.
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel last layer gradients to stabilize the training."""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution."""
        if isinstance(module, nn.Linear):
            _no_grad_trunc_normal_(module.weight, mean=0, std=0.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one forward pass through the head."""
        x = self.layers(x)
        # l2 normalization
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINO(torch.nn.Module):
    '''Main class for DINO.'''
    def __init__(
        self,
        name_model: str = 'resnet18',
        in_channels: int = 3,
        image_size: int = 256,
        patch_size: int = 16,
        drop_path_rate: float = 0.1,
        out_dim: int = 2048,
        hidden_dim: int = 512,
        bottleneck_dim: int = 64,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
        url_checkpoint: str = None
    ):
        '''
        input_dim (int): dimension of the output of the backbone
        '''
        super().__init__()

        # Load the backbone
        backbone = Backbone(name_model, in_channels)
        self.in_features = backbone.in_features
        self.out_dim = out_dim
        self.in_channels = in_channels

        # Define the student
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim=self.in_features,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=out_dim,
            freeze_last_layer=freeze_last_layer,
            norm_last_layer=norm_last_layer)

        # Define the teacher
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim=self.in_features,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=out_dim,
            freeze_last_layer=freeze_last_layer,
            norm_last_layer=True)

        if url_checkpoint:
            self.load_from_checkpoint(url_checkpoint)

        # Freeze teacher parameters
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def load_from_checkpoint(self, url_checkpoint: str):
        checkpoint = torch.load(url_checkpoint, weights_only=False, map_location='cpu')
        self.load_state_dict(checkpoint)

    def forward(self, x: torch.Tensor):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x: torch.Tensor):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z


class PreTrainedDINO(nn.Module):
    def __init__(self, url_checkpoint: str = None):
        '''
        Load DINO teacher model for evaluation.
        '''

        super().__init__()

        config_file = os.path.join(os.path.dirname(url_checkpoint), 'config.yaml')
        cfg = OmegaConf.load(config_file)

        # Use teacher backbone only
        self.backbone = DINO(
            name_model=cfg.student.arch,
            in_channels=cfg.student.in_channels,
            image_size=cfg.crops.global_crops_size,
            patch_size=cfg.student.patch_size,
            out_dim=cfg.student.out_dim,
            hidden_dim=cfg.student.hidden_dim,
            bottleneck_dim=cfg.student.bottleneck_dim,
            url_checkpoint=url_checkpoint
        ).teacher_backbone
        deactivate_requires_grad(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(start_dim=1)
        return x


if __name__ == '__main__':

    model = PreTrainedDINO(url_checkpoint='models_data/FINAL/dino_resnet50/checkpoint_164_linear=75.50_knn=73.15.pth.tar')
    model.eval()
    y = model.forward(torch.randn(1, 3, 256, 256))
    print(y.shape)
