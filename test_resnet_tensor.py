import os

import torch
from transformers import AutoModelForImageClassification

from models.resnet_tensor import ResNetForImageClassificationTensor


def test_tensor_output():
    # Load the original model
    original_model = AutoModelForImageClassification.from_pretrained(
        "galeio-research/nereus-sar-1-tengeop", token=os.environ["HF_TOKEN"]
    ).to("cuda")
    original_model.eval()

    # Create the tensor version of the model
    tensor_model = ResNetForImageClassificationTensor(original_model.config)
    tensor_model.load_state_dict(original_model.state_dict())
    tensor_model.to("cuda")
    tensor_model.eval()

    # Generate test input
    x = torch.randn(1, 1, 256, 256).cuda()

    # Get outputs from both models
    with torch.no_grad():
        original_output = original_model(x)
        tensor_output = tensor_model(x)

    # Print the shapes and types
    print("Original model output type:", type(original_output))
    print("Tensor model output type:", type(tensor_output))
    print("Tensor model output shape:", tensor_output.shape)

    # Verify the outputs are the same
    if isinstance(original_output, dict):
        original_tensor = original_output["logits"]
    else:
        original_tensor = original_output.logits

    print("Outputs are equal:", torch.allclose(original_tensor, tensor_output))


if __name__ == "__main__":
    test_tensor_output()
