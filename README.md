# MaizeNet-Precision

**MaizeNet-Precision** is a YOLOv8-based, domain-knowledge-guided framework for detecting maize seedlings in fixed-field monitoring images. It combines multiscale large separable kernel attention (**MLSKA**), hypergraph fusion (**HFM**, consisting of HCM and MANet), and **Powerful-IoU v2 (PIoU)**.

## Files

| File                              | Description |
|-----------------------------------| --- |
| `MLSKA.py`                        | MLSKA module with 7, 35, and 53 receptive-field branches. |
| `HFM.py`                          | HyperComputeModule (HCM) and MANet. |
| `loss.py`, `metrics.py`, `tal.py` | Modified Ultralytics files implementing PIoU v2 for box regression and target assignment. |
| `yolov8s_M_H_P.yaml`              | MaizeNet-Precision model architecture. |
| `yolov8_config_M_H_P.yaml`        | Training configuration. |
| `train_test.py`                   | Training and test-set evaluation. |
| `split.py`                        | Utility for splitting a YOLO-format dataset. |

## Setup

The experiments used **PyTorch 2.8.0 (CUDA 12.9)** and **Ultralytics 8.4.135**.

1. Install the corresponding PyTorch build and `ultralytics==8.3.167`.
2. Copy `MLSKA.py` and `HFM.py` into `ultralytics/nn/modules/`, and import their classes in `ultralytics/nn/tasks.py`:

   ```python
   from ultralytics.nn.modules.MLSKA import MLSKAP_K7_35_53
   from ultralytics.nn.modules.HFM import HyperComputeModule, MANet
   ```

3. Edit **`parse_model()`** in `ultralytics/nn/tasks.py` as follows.

   **(a) Read the hypergraph threshold from `scales`.** Replace the original scale-unpacking section at the beginning of `parse_model()` with:

   ```python
   if scales:
       scale = d.get("scale")
       if not scale:
           scale = tuple(scales.keys())[0]
           LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
       if len(scales[scale]) == 3:
           depth, width, max_channels = scales[scale]
       elif len(scales[scale]) == 4:
           depth, width, max_channels, threshold = scales[scale]
   ```

   For `yolov8s_M_H_P.yaml`, Ultralytics automatically selects the **`s`** scale, so `threshold = 8`; no explicit `scale: s` is needed.

   **(b) Add `MANet` to the existing `base_modules` set:**

   ```python
   base_modules = frozenset({
       # ... existing Ultralytics base modules ...
       MANet,
   })
   ```

   Keep `MANet` **out of `repeat_modules`** to match the supplied `tasks.py` implementation and its handling of the YAML repeat count.

   **(c) Add these branches to the main `if`/`elif` module-parsing chain**, after the `base_modules` branch and before the generic module cases:

   ```python
   elif m is MLSKAP_K7_35_53:
       c2 = ch[f]             # attention preserves the channel count
       args = [c2]           # MLSKAP_K7_35_53(dim=c2)

   elif m is HyperComputeModule:
       c1, c2 = ch[f], args[0]
       c2 = make_divisible(min(c2, max_channels) * width, 8)
       args = [c1, c2, threshold]  # HyperComputeModule(c1, c2, threshold)
   ```

   These branches use the class imports from Step 2. Keep the rest of `parse_model()` unchanged.
4. Back up and replace `ultralytics/utils/loss.py`, `metrics.py`, and `tal.py` with the supplied versions.

## Training and testing

Set the dataset path in `yolov8_config_M_H_P.yaml` (`data: ./data/corn.yaml`) and ensure its `model` path points to `yolov8s_M_H_P.yaml`. Also update the configuration path in `train_test.py` to match your local layout.

**Train:**

```bash
python train_test.py
```

**Evaluate:** after setting the trained `best.pt` path and dataset path in `test()`:

```bash
python -c "from train_test import test; test()"
```

The provided training configuration uses **1024 × 1024** input, **600 epochs**, **batch size 16**, and **seed 3407**. The evaluation function reports mAP50, mAP50–95, precision, and recall.



## Results

On **876 images** with **81,135 annotated seedlings** from five sites, MaizeNet-Precision achieved **97.00 ± 0.37% mAP50** and **64.50 ± 1.64% mAP50–95** (mean ± standard deviation over five random splits).

## Data 

The field-image dataset is available from the authors upon request.
