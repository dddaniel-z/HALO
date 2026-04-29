# HALO: Half-Frame-Rate Adaptive Learnable Operator for Lightweight STFT-Based Speech Enhancement

HALO is a causal plug-in module for lightweight STFT-based speech enhancement. It reduces overlap-induced temporal redundancy by running the enhancement backbone at a half internal frame rate and then restoring the enhanced spectrum to the original STFT grid. The original STFT/ISTFT procedure is preserved, and no additional algorithmic lookahead is introduced.

## Overview

STFT-based speech enhancement commonly uses overlapping analysis frames. While overlap is important for stable reconstruction, adjacent STFT frames can be highly correlated, leading to redundant computation in lightweight models.

HALO addresses this redundancy with two adaptive learnable operators:

- **Frame-rate reduction**: compresses adjacent causal STFT frames into a half-frame-rate representation before the backbone.
- **Frame-rate restoration**: reconstructs the full-rate complex spectrum after the backbone.
- **Causal plug-in design**: can be inserted into existing STFT-based enhancement backbones without changing the STFT/ISTFT pipeline.
- **Compute reallocation**: the reduced internal frame rate saves temporal computation, which can be used for channel widening under matched MAC/s comparisons.

## Repository Structure

```text
HALO/
├── code/
│   ├── cfg_train.toml        # Training configuration
│   ├── datasets.py           # Dataset loading and mixture preparation
│   ├── gtcrn.py              # Baseline GTCRN implementation
│   ├── gtcrn_halo.py         # GTCRN with HALO plug-in module
│   ├── loss.py               # Training loss functions
│   ├── score_utils.py        # Evaluation utilities
│   ├── train.py              # Training entry point
│   └── trainer.py            # Training loop
│
├── samples/
│   ├── sample1/
│   │   ├── sample1_wav/      # Waveform examples
│   │   └── sample1_spec/     # Spectrogram examples
│   ├── sample2/
│   ├── sample3/
│   └── sample4/
│
└── README.md
