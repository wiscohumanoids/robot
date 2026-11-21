"""Load the pretrained DINOv3 ViT-S/16 encoder used by the Panthera adapters."""

import timm
from huggingface_hub import hf_hub_download

MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
MODEL_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"


def load_backbone():
    """Load frozen ImageNet-normalized RGB features at a 16-pixel patch stride."""
    weights = hf_hub_download(
        repo_id=f"timm/{MODEL_NAME}",
        filename="model.safetensors",
        token=False,
        revision=MODEL_REVISION,
    )
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        checkpoint_path=weights,
        dynamic_img_size=True,
    )
    return model.requires_grad_(False).eval()
