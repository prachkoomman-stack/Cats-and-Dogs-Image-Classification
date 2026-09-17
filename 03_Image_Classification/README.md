---
title: Oxford Pets Breed Lens
emoji: 🐾
colorFrom: orange
colorTo: stone
sdk: gradio
app_file: app.py
pinned: false
---

# Oxford Pets Breed Lens

Upload a cat or dog image, choose one of the four local ResNet18 checkpoints, and compare the top five predicted breeds.

The checkpoints are trained on 37 Oxford-IIIT Pet breeds. The app uses each checkpoint's saved image size, normalization values, and class names at inference time.