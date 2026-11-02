# PU_RS PUGNet Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/PU_RS`  
Bi-temporal project: `building-change-detection/`  
Dataset root: `/home/xxj/Data`

## Clone and update

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test
git clone --branch reproduce https://github.com/xinjiaXu/PU_RS.git PU_RS
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/PU_RS
git switch reproduce
git pull --ff-only
```

## Environment and pretrained backbone

Create the isolated bi-temporal environment and check data paths:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/PU_RS
bash setup_server_env.sh
conda activate purs-bcd-rs
bash prepare_server_data.sh /home/xxj/Data
```

The PUGNet code defaults to PVT-v2-B4 and loads `building-change-detection/pretrained_model/pvt_v2_b4.pth`. The official README links a Baidu Disk PVT-v2 download (extraction code `dadp`); obtain the B4 checkpoint from that package and place it at the expected path. This weight is needed for both training and testing and is ignored by Git.

## Dataset layout

The scripts read the shared dataset folders directly. No image copies or list files are made.

| Key | Dataset root | Image A | Image B | Label |
| --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` |

The adapted loader accepts PNG/JPEG/TIFF files and black/white RGB masks. Training keeps the original mosaic augmentation and normalization.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

Defaults are the source training values: 200 epochs, batch size 8, input size 256, PVT-v2-B4 backbone, PUGNet channel width 32, Adam, lr `1e-4`, and decay 0.1 every 50 epochs. Each epoch validates on the `val` split and saves `Seg_epoch_best.pth` by validation F1. Select GPU 1 with `GPU_ID=1`; runtime overrides are `NUM_WORKERS=2` and `BATCH_SIZE=8`.

## Test

```bash
bash run_test.sh levircd
bash run_test.sh levircdplus
bash run_test.sh cdd
bash run_test.sh whucd
bash run_test.sh sysucd
```

The test wrapper loads the trained checkpoint for the chosen dataset, evaluates its `test` split, and saves metrics/prediction maps under `PU_RS_work_dirs/<key>/test_predictions/`.
