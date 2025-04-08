#!/usr/bin/env python3

import sys

import torch
import torch.nn as nn
import torchvision


def prepare_model():
    resnet50 = torchvision.models.resnet50(weights=None)

    # replace the first conv layer to accept single channel input
    resnet50.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=resnet50.conv1.out_channels,
        kernel_size=resnet50.conv1.kernel_size,
        stride=resnet50.conv1.stride,
        padding=resnet50.conv1.padding,
        bias=(resnet50.conv1.bias is not None),
    )

    # remove the final fc layer
    resnet50.fc = nn.Identity()

    return resnet50


def main(path):
    resnet50 = prepare_model()

    state_dict = torch.load(path, weights_only=True)
    resnet50.load_state_dict(state_dict)


if __name__ == "__main__":
    path_ckpt = sys.argv[1]

    main(path_ckpt)
