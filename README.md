# Pointcept with PointCNN++

This repository extends Pointcept with a PointCNN++ backbone and supports **MSC (Masked Scene Contrast) pretraining** and **semantic segmentation fine-tuning** on LiDAR data such as NuScenes. This document describes how to install dependencies, prepare data, and run PointCNN++ pretraining and fine-tuning.

---

## Dependencies

### 1. Environment and base requirements

- **Python**: 3.8-3.10 recommended.
- **CUDA**: Match your PyTorch version (e.g. CUDA 12.1).
- **Optional Conda packages** (install if needed):
  ```bash
  conda install cuda cudnn gcc=13.2 gxx=13.2 ninja google-sparsehash
  ```

### 2. Install PyTorch and project dependencies

From the project root:

```bash
cd /path/to/Pointcept
pip install -r requirements.txt
```

`requirements.txt` includes `h5py`, `tensorboard`, `wandb`, `open3d`, etc.

### 3. PointCNN++ installation

PointCNN++ dependencies must be installed separately (used by the PointCNN++ backbone for pretraining and fine-tuning):

### 4. Local operator libraries (optional)

If you use PointOps / PointGroup operators:

```bash
# after activating your conda environment
bash libs/install_pointops.sh
```

---

## Steps overview

1. Install the dependencies above (including PointCNN++).
2. Prepare NuScenes preprocessed data and, if needed, create the `nuscenes_pretrain` / `nuscenes_finetune` config symlinks.
3. Run PointCNN++ pretraining (MSC).
4. Run NuScenes semantic segmentation fine-tuning using the pretrained weights.
5. (Optional) Evaluate on validation/test sets.

---

## Data preparation

Pretraining and fine-tuning both use **preprocessed** NuScenes under a single `data_root` (e.g. `data/nuscenes_processed`). Fine-tuning scripts pass it via `--options data.train.data_root=...` etc.

**How to get `nuscenes_processed` from raw `nuscenes`:**

1. **Raw data**: Put the official NuScenes dataset at `data/nuscenes` (with `v1.0-trainval`, `v1.0-test`, `samples`, `sweeps`, `lidarseg`, etc.).

2. **Directory layout**: Create `data/nuscenes_processed` and make the raw NuScenes data visible under it. `NuScenesDataset` looks for files under `data_root/raw` first, so the usual approach is to symlink the raw dataset to `nuscenes_processed/raw`:

   ```bash
   mkdir -p /path/to/Pointcept/data/nuscenes_processed
   ln -s /path/to/Pointcept/data/nuscenes /path/to/Pointcept/data/nuscenes_processed/raw
   ```

   Alternatively, copy or symlink the contents of `data/nuscenes` directly into `data/nuscenes_processed` (so that `nuscenes_processed/v1.0-trainval/...` etc. exist).

3. **Generate info files**: Run the preprocessing script (requires the [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) package: `pip install nuscenes-devkit`):

   ```bash
   python pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py \
     --dataset_root /path/to/Pointcept/data/nuscenes \
     --output_root /path/to/Pointcept/data/nuscenes_processed \
     --max_sweeps 10
   ```

   This creates `data/nuscenes_processed/info/nuscenes_infos_10sweeps_{train,val,test}.pkl`. Use the same `data_root` (e.g. `data/nuscenes_processed`) in your config or script options.

---

## PointCNN++ pretraining (MSC)

Uses **Masked Scene Contrast** for self-supervised pretraining on NuScenes with a PointCNN++ backbone (ResUNetPointCNNpp).

### Custom arguments (using `scripts/train.sh`)

You can run pretraining with the generic script `scripts/train.sh` and the same config/experiment. Ensure the config path exists.

```bash
./scripts/train.sh \
  -c pretrain-msc-pointcnnpp-base \
  -n pointcnnpp_nuscenes_pretrain \
  -g 8
```

- `-d`: Dataset name (e.g. `nuscenes_pretrain`)
- `-c`: Config name (no `.py`)
- `-n`: Experiment name → output dir `exp/nuscenes_pretrain/<EXP_NAME>/`
- `-g`: Number of GPUs
- `-w`: Checkpoint path (for resume or warm start)
- `-r true`: Resume from last checkpoint (uses that experiment’s `config.py` and `model_last.pth`)

Pretrained weights are saved under:  
`exp/nuscenes_pretrain/<EXP_NAME>/model/model_last.pth` (and any best checkpoint per config).

---

## PointCNN++ fine-tuning (NuScenes semantic segmentation)

Fine-tune the pretrained PointCNN++ model for NuScenes semantic segmentation using `scripts/train.sh` with a finetune config and pretrained weight.

### 1. Set the pretrained weight path

Pass the pretrained checkpoint via `-w`:

```bash
./scripts/train.sh -c semseg-pointcnnpp-base \
  -n semseg_from_msc_pointcnnpp_pretrain -w /path/to/exp/nuscenes_pretrain/<EXP_NAME>/model/model_last.pth -g 8
```

### 2. Data path

`train.sh` does not override the data root; it is taken from the config. Set `data_root` in the finetune config (e.g. `configs/nuscenes/semseg-pointcnnpp-base.py`) to your preprocessed path (e.g. `data/nuscenes_processed` or an absolute path), or ensure the config’s `nuscenes` symlink points to `configs/nuscenes/finetune`.

### 3. Run fine-tuning

From the project root:

```bash
./scripts/train.sh \
  -d nuscenes \
  -c semseg-pointcnnpp-base \
  -n semseg_pointcnnpp_finetune \
  -w  None \
  -g 4
```

After fine-tuning, weights are typically under:  
`exp/nuscenes_finetune/<EXP_NAME>/model/model_last.pth`, `model_best.pth`, etc.

---

## Test / evaluation

Evaluate a fine-tuned PointCNN++ model on NuScenes validation/test using `scripts/test.sh`:

```bash
./scripts/test.sh \
  -d nuscenes_finetune \
  -n semseg_from_msc_pointcnnpp_pretrain \
  -w model_best \
  -g 1
```

- `-n`: Experiment name from fine-tuning (matches `exp/nuscenes_finetune/<EXP_NAME>`).
- `-w`: Weight filename without `.pth` (e.g. `model_best` or `model_last`).
- Data path is again set by the script’s `DATA_ROOT` or `--options data.*.data_root=...`; keep it consistent with fine-tuning.

Use `-f` to run in the foreground without nohup if the script supports it.

---

## Config and script mapping

| Stage      | Script            | Dataset (`-d`)     | Config |
|-----------|-------------------|--------------------|--------|
| Pretrain  | `scripts/train.sh` | `nuscenes_pretrain` | `pretrain-msc-pointcnnpp-v1m2-0-lidar-csc-enhanced` (under `configs/nuscenes/pretrain/`) |
| Fine-tune | `scripts/train.sh` | `nuscenes_finetune` | `semseg-pointcnnpp-base` (under `configs/nuscenes/finetune/`) |
| Test      | `scripts/test.sh`  | `nuscenes_finetune` | Uses `config.py` in the experiment dir or `-c` |

---
## Supported models and tasks

- **Pretraining**: MSC v1m2 + PointCNN++ (ResUNetPointCNNpp), config: `configs/nuscenes/pretrain/pretrain-msc-pointcnnpp-v1m2-0-lidar-csc-enhanced.py`.
- **Fine-tuning**: NuScenes 16-class semantic segmentation with CE + Focal + Lovász losses by default, config: `configs/nuscenes/finetune/semseg-pointcnnpp-base.py`.

To change backbone, losses, or data augmentation, edit the corresponding config or duplicate it and pass the new config name with `-c`.
