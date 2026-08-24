# Datasets

None of the four datasets are redistributed in this repo — each has its own license/access terms.
Download each from its official source below, then arrange it into the directory layout each dataset
class expects (see `promisenet/data/datasets.py`).

## ISIC-2017 (Skin Lesion)

- Source: [ISIC Challenge Archive, 2017](https://challenge.isic-archive.com/data/#2017)
- 2000 training / 150 validation / 600 test dermoscopic images with binary lesion masks.
- **Not offline-augmented** (per the paper's training protocol — this is the one dataset trained
  directly on the raw split).
- Expected layout (matches `ISICDataset` / `train.py --config configs/isic2017.yaml`):
  ```
  data_dir/
    training_images/*.jpg   training_masks/*_segmentation.png
    val_images/*.jpg        val_masks/*_segmentation.png
    test_images/*.jpg       test_masks/*_segmentation.png
  ```

## Kvasir-Polyp (Kvasir-SEG)

- Source: [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/)
- 1000 colonoscopy frames with polyp masks (800 train / 100 val / 100 test, per the paper).
- Offline-augmented (horizontal/vertical flip + rotation up to 360°) — after splitting into
  train/val/test yourself, run:
  ```
  python scripts/prepare_kvasir.py --raw_dir /path/to/split/kvasir_seg --out_dir ./data/kvasir_polyp_augmented --dataset_name kvasir_polyp
  ```
- Expected layout for `train.py --config configs/kvasir_polyp.yaml` (produced by the script above):
  ```
  data_dir/
    train/images/{original,H_flipped,V_flipped,rotated,combined}/*.png
    train/masks/{original,H_flipped,V_flipped,rotated,combined}/*.png
    val/images/...   val/masks/...
    test/images/...  test/masks/...
  ```

## Kvasir-Instrument

- Source: [datasets.simula.no/kvasir-instrument](https://datasets.simula.no/kvasir-instrument/)
- Endoscopic surgical-instrument images (472 train / 59 val / 59 test, per the paper).
- Offline-augmented, same pipeline and expected layout as Kvasir-Polyp above:
  ```
  python scripts/prepare_kvasir.py --raw_dir /path/to/split/kvasir_instrument --out_dir ./data/kvasir_instrument_augmented --dataset_name kvasir_instrument
  ```

## CAMUS (2D Echocardiography)

- Source: [CREATIS CAMUS Challenge](https://www.creatis.insa-lyon.fr/Challenge/camus/) — **licensed
  data, registration required; do not redistribute.**
- 400 patients train / 50 val / 50 test, multi-class (background + LV + MYO + LA).
- Offline-augmented:
  ```
  python scripts/prepare_camus.py --raw_dir /path/to/preseparated_camus --out_dir ./data/camus_augmented
  ```
- Expected layout matches Kvasir's above, just with 4-class masks instead of binary.

## Citations

Please cite the original dataset papers alongside this work:

- ISIC-2017: Codella et al., "Skin Lesion Analysis Toward Melanoma Detection," 2018.
- Kvasir-SEG: Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset," MMM 2020.
- Kvasir-Instrument: Jha et al., "Kvasir-Instrument: Diagnostic and Therapeutic Tool Segmentation
  Dataset in Gastrointestinal Endoscopy," MMM 2021.
- CAMUS: Leclerc et al., "Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D
  Echocardiography," IEEE TMI, 2019.
