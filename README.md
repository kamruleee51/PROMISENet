# PROMISE-Net

Official code for **PROMISE-Net** (**PRO**mpt **M**odulated **I**ntegration for **SE**gmentation),
the model proposed in *"Prompt-Conditioned Channel Attention for Hierarchical Feature Modulation
toward Anatomy-Agnostic Segmentation."*

This repo contains **only the proposed model** — PCCA (Prompt-Conditioned Channel Attention) and its
two backbones, PROMISE-CNN (U-Net) and PROMISE-Txformer (a custom transformer encoder + CNN decoder)
— trainable and testable on all four benchmarks from the paper: ISIC-2017, Kvasir-Polyp,
Kvasir-Instrument, and CAMUS. Baseline models (U-Net, UNETR, SAM) and ablation code are intentionally
not included here; this repo is scoped for reproducing the proposed model's results specifically.

## What is PCCA?

PCCA is a squeeze-and-excitation–style gating mechanism that conditions channel-wise feature
modulation on a user-provided bounding-box prompt: pooled image-feature and pooled prompt-feature
descriptors are projected into a shared latent space, summed, and passed through a small MLP + sigmoid
to produce a channel-wise gate — combined with a direct prompt-multiplication term for
spatially-precise modulation. See `promisenet/models/pcca.py`.

**Where PCCA is applied differs by backbone** — this is intentional, not an inconsistency:
- **PROMISE-CNN**: every encoder stage, the bottleneck, and every decoder stage.
- **PROMISE-Txformer**: the bottleneck and every decoder stage only. The encoder is the transformer
  itself (patch embedding + 12 self-attention layers) — there is no spatial feature map to gate until
  its output is reshaped at the bottleneck, so no encoder-stage PCCA is applied for this variant.

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

See [docs/DATASETS.md](docs/DATASETS.md) for download links, licensing notes, expected directory
layout, and the offline-augmentation preparation scripts for CAMUS/Kvasir-Polyp/Kvasir-Instrument
(ISIC-2017 is trained on the raw split, no augmentation, per the paper's protocol).

## Training

```bash
python train.py --config configs/isic2017.yaml         --model cnn         --data_dir ./data/isic2017
python train.py --config configs/isic2017.yaml         --model transformer --data_dir ./data/isic2017
python train.py --config configs/kvasir_polyp.yaml      --model cnn         --data_dir ./data/kvasir_polyp_augmented
python train.py --config configs/kvasir_instrument.yaml --model cnn         --data_dir ./data/kvasir_instrument_augmented
python train.py --config configs/camus.yaml             --model transformer --data_dir ./data/camus_augmented
```

`--model cnn` trains PROMISE-CNN; `--model transformer` trains PROMISE-Txformer. Each `configs/*.yaml`
encodes the exact per-dataset hyperparameters (loss weights, augmentation) verified against the
original experiments — see "Faithfulness notes" below before changing them.

## Testing

```bash
python test.py --config configs/isic2017.yaml --model cnn \
    --data_dir ./data/isic2017 --weights outputs/isic2017_cnn_best.pth --out_dir outputs/isic2017_cnn_test
```

Writes per-image DSC/IoU/HD95/FNR/FPR to `metrics.csv` (+ a mean/std `metrics_summary.csv`) and
predicted masks to `predictions/`.

## Project structure

```
PROMISENet/
├── promisenet/
│   ├── models/
│   │   ├── pcca.py                 PCCA module + bounding-box prompt encoder
│   │   ├── promisenet_cnn.py       PROMISE-CNN (U-Net + PCCA at every stage)
│   │   └── promisenet_txformer.py  PROMISE-Txformer (custom transformer + CNN decoder, PCCA at
│   │                               bottleneck + decoder only)
│   ├── data/
│   │   ├── datasets.py             ISICDataset, CAMUSDataset, KvasirDataset
│   │   └── bbox_utils.py           Automated bounding-box extraction from masks
│   ├── losses.py                   Dice/CE loss, per-dataset weighting
│   └── metrics.py                  DSC/IoU/HD95/FNR/FPR for evaluation
├── scripts/
│   ├── augment_utils.py            Offline augmentation (H/V flip, rotate, combined)
│   ├── prepare_camus.py            CAMUS augmentation CLI
│   └── prepare_kvasir.py           Kvasir-Polyp / Kvasir-Instrument augmentation CLI
├── configs/                        Per-dataset hyperparameters (one YAML each)
├── docs/DATASETS.md
├── train.py
└── test.py
```

## Faithfulness notes (read before changing hyperparameters)

This code is a direct extraction from the original experiment notebooks. Where a canonical source was
ambiguous, it was resolved either by matching candidate implementations against the paper's exact
reported numbers, or by direct confirmation from the paper's author — see
`promisenet/models/promisenet_txformer.py`'s module docstring for one case where those two methods
initially disagreed, and the author's confirmation of which implementation to use. Non-obvious
properties of the *actual* training code are preserved here rather than "corrected," because
correcting them would mean the code no longer reproduces the paper's reported results:

1. **PCCA's placement differs by backbone** — PROMISE-CNN gates every encoder stage, the bottleneck,
   and every decoder stage; PROMISE-Txformer gates only the bottleneck and every decoder stage (no
   encoder-stage gating, since the encoder is the transformer itself). See "What is PCCA?" above and
   `promisenet/models/promisenet_txformer.py`'s module docstring.
2. **Dice loss is non-differentiable everywhere.** Both models, on all four datasets, compute the
   Dice-loss term from `argmax`-then-one-hot predictions, which has no real gradient path back to the
   model — training is effectively driven by the cross-entropy term alone, despite the code nominally
   computing "Dice + CE". See `promisenet/losses.py`'s module docstring.
3. **Both models' cross-entropy term is computed on already-softmaxed values** — both `forward()`
   methods apply softmax internally, and `compute_loss` must be called with that output unchanged (no
   extra softmax in between). This is mathematically unusual (a second softmax gets applied inside
   `nn.CrossEntropyLoss`) but is what both models were actually trained with.
4. **ISIC-2017's automated bounding box differs between the two models**: PROMISE-CNN expands the
   ground-truth box by a fixed ±20px; PROMISE-Txformer expands it by a random [0,20)px offset,
   redrawn every call. See `promisenet/data/bbox_utils.py`'s module docstring and
   `configs/isic2017.yaml`.

## Citation

```bibtex
@article{promisenet,
  title   = {Prompt-Conditioned Channel Attention for Hierarchical Feature Modulation toward Anatomy-Agnostic Segmentation},
  author  = {Hossain, Mosharof and Islam, Md Rabiul and Halder, Limon and Serpedin, Erchin and Hasan, Md Kamrul},
}
```

## License

See [LICENSE](LICENSE).
