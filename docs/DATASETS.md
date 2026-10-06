# Datasets

None of the four datasets are redistributed in this repo — each has its own license/access terms.

**Recommended (quick start): pre-cleaned Kaggle mirrors.** The author already maintains resized/
augmented versions of all four datasets on Kaggle, ready to use directly with the paths below — no
need to run `scripts/prepare_camus.py` / `scripts/prepare_kvasir.py` yourself. If you'd rather build
the data from scratch (e.g. to audit the exact preprocessing, or because you don't have Kaggle access),
each dataset's official source and the prep scripts are listed as the alternative under each heading.

> The paths below assume the layout `augment_split()` (`scripts/augment_utils.py`) produces —
> `{split}/images/{aug_type}/*.png`, mirrored under `masks/`. If your download differs, adjust the
> glob patterns in `train.py`/`test.py` or reorganize the files to match.

## ISIC-2017 (Skin Lesion)

- **Recommended**: [kaggle.com/datasets/mosharofhossain/isic-2017-skin-lesion-segmentation-dataset](https://www.kaggle.com/datasets/mosharofhossain/isic-2017-skin-lesion-segmentation-dataset)
- Alternative (official, unprocessed): [ISIC Challenge Archive, 2017](https://challenge.isic-archive.com/data/#2017)
- 2000 training / 150 validation / 600 test dermoscopic images with binary lesion masks.
- **Not offline-augmented** (per the paper's training protocol — this is the one dataset trained
  directly on the raw split; `ISICDataset` resizes to 256x256 on the fly, no prep script needed either
  way).
- Expected layout (matches `ISICDataset` / `train.py --config configs/isic2017.yaml`):
  ```
  data_dir/
    training_images/*.jpg   training_masks/*_segmentation.png
    val_images/*.jpg        val_masks/*_segmentation.png
    test_images/*.jpg       test_masks/*_segmentation.png
  ```

## Kvasir-Polyp (Kvasir-SEG)

- **Recommended**: [kaggle.com/datasets/metaloxide24/kvasir-with-augmented-instances](https://www.kaggle.com/datasets/metaloxide24/kvasir-with-augmented-instances)
  (256x256, confirmed — this is the dataset actually used; a separate 352x352 mirror also exists but
  does **not** match the resolution used by either canonical PROMISE-CNN/PROMISE-Txformer training
  notebook, both of which hardcode `img_size=256` — use the 256px version for reproducing the paper's
  reported numbers).
- Alternative (official, unprocessed): [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/)
- 1000 colonoscopy frames with polyp masks (800 train / 100 val / 100 test, per the paper).
- Offline-augmented (horizontal/vertical flip + rotation up to 360°). To build from scratch instead of
  using the Kaggle mirror: split into train/val/test yourself, then run:
  ```
  python scripts/prepare_kvasir.py --raw_dir /path/to/split/kvasir_seg --out_dir ./data/kvasir_polyp_augmented --dataset_name kvasir_polyp
  ```
- Expected layout for `train.py --config configs/kvasir_polyp.yaml`:
  ```
  data_dir/
    train/images/{original,H_flipped,V_flipped,rotated,combined}/*.png
    train/masks/{original,H_flipped,V_flipped,rotated,combined}/*.png
    val/images/...   val/masks/...
    test/images/...  test/masks/...
  ```

## Kvasir-Instrument

- **Recommended**: [kaggle.com/datasets/mosharofhossain/kvasir-instrument-resized-256-augmented](https://www.kaggle.com/datasets/mosharofhossain/kvasir-instrument-resized-256-augmented)
- Alternative (official, unprocessed): [datasets.simula.no/kvasir-instrument](https://datasets.simula.no/kvasir-instrument/)
- Endoscopic surgical-instrument images (472 train / 59 val / 59 test, per the paper).
- Offline-augmented (per the paper's protocol — the "256" in the Kaggle dataset name matches
  `configs/kvasir_instrument.yaml`'s `img_size: 256`). To build from scratch instead:
  ```
  python scripts/prepare_kvasir.py --raw_dir /path/to/split/kvasir_instrument --out_dir ./data/kvasir_instrument_augmented --dataset_name kvasir_instrument
  ```
- Expected layout: same as Kvasir-Polyp above.

## CAMUS (2D Echocardiography)

- **Recommended**: [kaggle.com/datasets/mosharofhossain/camus-segmentation-augmented-resized-256](https://www.kaggle.com/datasets/mosharofhossain/camus-segmentation-augmented-resized-256)
- Alternative (official, unprocessed, **licensed — registration required, do not redistribute**):
  [CREATIS CAMUS Challenge](https://www.creatis.insa-lyon.fr/Challenge/camus/)
- 400 patients train / 50 val / 50 test, multi-class (background + LV + MYO + LA).
- Offline-augmented (the "256" in the Kaggle dataset name matches `configs/camus.yaml`'s
  `img_size: 256`). To build from scratch instead:
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
