# PROMISE-Net

Official code for **PROMISE-Net** (**PRO**mpt **M**odulated **I**ntegration for **SE**gmentation),
the model proposed in *"Prompt-Conditioned Channel Attention for Hierarchical Feature Modulation
toward Anatomy-Agnostic Segmentation."*

This repo contains **only the proposed model** — PCCA (Prompt-Conditioned Channel Attention) and its
two backbones, PROMISE-CNN (U-Net) and PROMISE-Txformer (UNETR) — trainable and testable on all four
benchmarks from the paper: ISIC-2017, Kvasir-Polyp, Kvasir-Instrument, and CAMUS. Baseline models
(U-Net, UNETR, SAM) and ablation code are intentionally not included here; this repo is scoped for
reproducing the proposed model's results specifically.

## What is PCCA?

PCCA is a squeeze-and-excitation–style gating mechanism that conditions channel-wise feature
modulation on a user-provided bounding-box prompt: pooled image-feature and pooled prompt-feature
descriptors are projected into a shared latent space, summed, and passed through a small MLP + sigmoid
to produce a channel-wise gate — combined with a direct prompt-multiplication term for
spatially-precise modulation. See `promisenet/models/pcca.py`.

PCCA is applied at every encoder stage and the bottleneck for both backbones. For PROMISE-CNN, it is
also re-applied at every decoder stage. For PROMISE-Txformer, "encoder stage" means the skip-connection
features UNETR builds from intermediate ViT hidden states (not the transformer's self-attention
computation itself, which is never conditioned on the prompt) — see
`promisenet/models/promisenet_txformer.py`'s `forward()`.

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
│   │   └── promisenet_txformer.py  PROMISE-Txformer (UNETR + PCCA)
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

## Notes (read before changing hyperparameters)

This code is a direct extraction from the original experiment notebooks. A few non-obvious properties
of the training setup are preserved exactly, since changing them would mean the code no longer
reproduces the paper's reported results:

1. **Dice-loss differentiability varies by (model, dataset)** — PROMISE-CNN (all datasets) and
   PROMISE-Txformer on CAMUS compute the Dice-loss term from `argmax`-then-one-hot predictions, which
   has no real gradient path back to the model; training is effectively driven by the cross-entropy
   term alone in these cases. PROMISE-Txformer on the three binary datasets computes Dice from softmax
   probabilities (properly differentiable). See `promisenet/losses.py`.
2. **PROMISE-CNN's cross-entropy term is computed on already-softmaxed values** — its `forward()`
   applies softmax internally, and `compute_loss` must be called with that output unchanged (no extra
   softmax in between). PROMISE-Txformer returns raw logits instead.
3. **ISIC-2017's automated bounding box differs between the two models**: PROMISE-CNN expands the
   ground-truth box by a fixed ±20px; PROMISE-Txformer expands it by a random [0,20)px offset,
   redrawn every call. See `promisenet/data/bbox_utils.py` and `configs/isic2017.yaml`.

## Citation

```bibtex
@article{promisenet,
  title   = {Prompt-Conditioned Channel Attention for Hierarchical Feature Modulation toward Anatomy-Agnostic Segmentation},
  author  = {Hossain, Mosharof and Islam, Md Rabiul and Halder, Limon and Serpedin, Erchin and Hasan, Md Kamrul},
}
```

## License

See [LICENSE](LICENSE).
