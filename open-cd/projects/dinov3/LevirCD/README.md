# DINOv3 on Open-CD LEVIR-CD

This project adds a DINOv3 ViT-L/16 backbone for Open-CD change detection.
It uses Open-CD's `SiamEncoderDecoder`: the same frozen DINOv3 backbone encodes
the pre- and post-event RGB images, `DINOv3FeatureFusionPyramid` fuses the
features with absolute difference, and UPerNet predicts the binary change map.

Update the pretrained checkpoint path in:

```text
projects/dinov3/configs/levir_cd/dinov3-vitl16_upernet_4xb2-40k_levircd-512x512.py
```

Then train with:

```bash
python tools/train.py \
  projects/dinov3/configs/levir_cd/dinov3-vitl16_upernet_4xb2-40k_levircd-512x512.py
```

The config expects the LEVIR-CD layout used by the existing Open-CD configs:

```text
data/LEVIR-CD/train/{A,B,label}
data/LEVIR-CD/val/{A,B,label}
data/LEVIR-CD/test/{A,B,label}
```

