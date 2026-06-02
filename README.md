<p align="center">
  <h1 align="center">RhymeFlow: Training-Free Acceleration for Video Generation with Asynchronous Denoising Flow Scheduling</h1>
  <h3 align="center"><a href="https://arxiv.org/abs/2604.08370">Paper</a> | <a href="https://drive.google.com/file/d/11m6sbfPQDMKlIYYH2Z628UjGO1ASPuLm/view?usp=sharing">Project Page</a></h3>
</p>

## TODO List

- ✅ Release the Wan 2.1 adaptation.
- 🚧 Release the HunyuanVideo adaptation.

## Installation

Begin by cloning the repository:

```bash
git clone https://github.com/Simon-Dcs/RhymeFlow.git
cd RhymeFlow
```

We recommend using CUDA versions 12.4 / 12.8 + PyTorch versions 2.5.1 / 2.6.0

```bash
# 1. Create and activate conda environment
conda create -n rhymeflow python==3.12.9 # or 3.11.9 if have error when installing kernels
conda activate rhymeflow

# 2. Install uv, then install other packages
pip install uv
uv pip install -e .

pip install flash-attn --no-build-isolation

# 4. Install customized kernels. (You might need to upgrade your cmake and CUDA version.)
pip install -U setuptools # Require at least version 77.0.0
git submodule update --init --recursive
cd svg/kernels
pip install -U cmake
bash setup.sh

# 5. Install FlashInfer (standard) and cuVS
cd 3rdparty/flashinfer
pip install --no-build-isolation --verbose --editable .
pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com

# Optional: If the FlashInfer monkey patch fails in your environment,
# install the manually patched FlashInfer (block sparse with varied block sizes).
cd 3rdparty/flashinfer
cp ../../../../assets/patches/modifications.patch ./
git apply modifications.patch
pip install --no-build-isolation --verbose --editable . # Block Sparse Attention with varied block sizes
```


## Quick Start

Run one prompt and one method:

```bash
GPU_ID=0 PROMPT_ID=3 METHOD=rhyme bash scripts/wan/wan_t2v_case.sh
```

Available `METHOD` values:

| Method | Output dir | Default parameters |
|---|---|---|
| `dense` | `dense` | Full Wan2.1 denoising |
| `svg` | `svg_s03` | `sparsity=0.3`, `num_sampled_rows=64` |
| `sap` | `sap_default_q300_k800_tp092` | `q=300,k=800,top_p=0.92,min_kc_ratio=0.10,iter=50/2` |
| `rhyme` | `rhyme_tw10_m2_skip3-5` | `Tw=10,M=2,skip=3-5,semantic keyframes,scheduler_approx` |
| `rhyme_sap` | `rhyme_sap_tw8_m3_skip3-5_q350_k1200_tp098_min020_it5` | `Tw=8,M=3,skip=3-5` plus `q=350,k=1200,top_p=0.98,min_kc_ratio=0.20,iter=5/2` |
 

## Batch Reproduction

Run the default prompts:

```bash
GPU_IDS="0 1 2 3" bash scripts/wan/wan_t2v_batch.sh
```

The default prompt list is:

```text
1 2 3 4 5 6
```

You can override the method set:

```bash
METHODS="dense sap rhyme rhyme_sap" GPU_IDS="0 1 2 3" bash scripts/wan/wan_t2v_batch.sh
```

## Citation

```bibtex
```

## License

See `LICENSE.txt`.
