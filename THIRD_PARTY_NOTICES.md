# Third-party materials and attribution

Original project wrappers, tests and documentation use the Apache License 2.0 in [LICENSE](LICENSE). NVIDIA model weights and official sample materials are not relicensed under Apache-2.0.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Official model cards declare OpenMDW License Agreement version 1.1. A copy is retained in [licenses/NVIDIA-OpenMDW-1.1.txt](licenses/NVIDIA-OpenMDW-1.1.txt).

| Material | Source / pinned revision | Published derivatives |
| --- | --- | --- |
| Cosmos3-Nano-Sim-Bimanual official examples | [NVIDIA Hugging Face repository](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/da4d2c09175e2a6dd5b237265318066d5134312e) | `media/bimanual-inputs.jpg`, `media/action-trajectory.jpg`, `media/robot-input.jpg` |
| Cosmos3-Nano-Sim-Transfer official example2 | [NVIDIA Hugging Face repository](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/45510d5b8cefa338b7abc01eb702cf1e9cdf6742) | `media/transfer-inputs.jpg` |
| Official Cosmos Framework runtime, documentation and input validator | [NVIDIA GitHub repository](https://github.com/NVIDIA/cosmos-framework/tree/bbd4a67c3cf54cc6699464ad1259f3a0f6537da9) | Referenced/invoked from a local checkout; upstream code is not vendored into this repository |

Preview modifications consist of resizing, video frame selection, labels, contact sheets, and plotting already-normalized action values. They are labeled as official inputs, not locally generated model outputs. Model weights, full official recordings and upstream source trees are excluded from Git.

The pinned Nano text tokenizer and Wan VAE are separate dependencies; consult their original licenses before use or redistribution. No checkpoint or gated credentials are distributed by this repository. This is an independent reproduction project and does not imply NVIDIA endorsement.
