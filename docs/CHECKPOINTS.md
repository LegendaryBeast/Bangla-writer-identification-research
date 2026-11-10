# Exact model checkpoint references

No network is trained or fine-tuned on the study writers. Pretrained weights are not redistributed. Numerical reproduction uses the supplied frozen features and does not download weights.

| Representation | torchvision 0.29.0 enum | Official checkpoint | Filename SHA-256 prefix |
|---|---|---|---|
| ResNet-18 and binary Stroke-CNN | `ResNet18_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/resnet18-f37072fd.pth | `f37072fd` |
| EfficientNet-B0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth | `7f5810bc` |
| ViT-B/16 | `ViT_B_16_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/vit_b_16-c867db91.pth | `c867db91` |

These URLs and enum names were verified against the installed torchvision 0.29.0 package. The suffixes above are the official filename hash prefixes, not claims of full SHA-256 digests. The originally generated feature matrices are independently covered by the release's complete SHA-256 manifest. `manifests/checkpoints.json` is machine readable.

Feature extraction uses torch 2.14.0 and the custom image preparation in `wid_features.py`, not an assumed default enum transform. Official architecture documentation: https://docs.pytorch.org/vision/stable/models.html
