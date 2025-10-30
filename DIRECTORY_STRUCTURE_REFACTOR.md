# Directory Structure Refactoring Guide

## Overview
Successfully refactored the entire data collection and model training pipeline to use a unified parent data folder structure. This eliminates code duplication and makes it easier to manage multiple datasets.

---

## New Directory Structure

```
trento_house_data/              # Parent data folder (configurable)
├── train/                      # All training images (flat, no labels)
│   ├── image_001.png
│   ├── image_002.jpg
│   └── ...
└── test/                       # Test data organized by label
    ├── normal/                 # Normal test samples
    │   ├── normal_001.png
    │   ├── normal_002.jpg
    │   └── ...
    └── anomalous/              # Anomalous test samples
        ├── anomaly_001.png
        ├── anomaly_002.jpg
        └── ...
```

### Previous Structure (Deprecated)
```
training_data/                  # All training images
testing_data/                   # Test root
├── normal/
└── anomalous/
```

---

## Updated Files & Changes

### 1. `data_collect.py`
**Purpose**: Collect images from camera/API and organize into new directory structure.

**Changes**:
- `--out` flag now takes parent data folder path (default: `trento_house_data`)
- Internally constructs:
  - Train path: `{data_parent}/train/`
  - Test path: `{data_parent}/test/normal/` for normal samples

**Usage**:
```bash
# Collect training images
python data_collect.py --mode train --out trento_house_data
# Saves to: trento_house_data/train/

# Collect test images (normal samples)
python data_collect.py --mode test --out trento_house_data
# Saves to: trento_house_data/test/normal/

# Custom data folder
python data_collect.py --mode train --out my_custom_data
# Saves to: my_custom_data/train/
```

### 2. `train.py` (Main Training Script)
**Class**: `TrainingOrchestrator`

**Method Changed**: `run()`

**Old Signature**:
```python
def run(
    self,
    configs: List[Dict[str, Any]],
    train_folder: str = "training_data",
    test_folder: str = "testing_data",
    weights_folder: str = "./weights",
) -> List[TrainingResult]:
```

**New Signature**:
```python
def run(
    self,
    configs: List[Dict[str, Any]],
    data_parent: str = "trento_house_data",
    weights_folder: str = "./weights",
) -> List[TrainingResult]:
```

**How It Works**:
```python
# Internally constructs paths:
data_parent_path = Path(data_parent)
train_folder = str(data_parent_path / "train")           # {data_parent}/train
test_folder = str(data_parent_path / "test")             # {data_parent}/test
```

**Usage**:
```python
orchestrator = TrainingOrchestrator(device, logger)
results = orchestrator.run(
    configs=configs,
    data_parent="trento_house_data"  # or any other parent folder
)
```

### 3. `models/model.py` - `AnomalyDetectionModel`
**Method Changed**: `test_folder()`

**Old Signature**:
```python
def test_folder(
    self,
    test_folder: Union[str, Path],
    output_folder: Union[str, Path] = "test_results",
) -> Dict[str, Any]:
```

**New Signature**:
```python
def test_folder(
    self,
    data_parent: Union[str, Path],
    output_folder: Union[str, Path] = "test_results",
) -> Dict[str, Any]:
```

**How It Works**:
```python
# Internally constructs test path:
data_parent = Path(data_parent)
test_folder = data_parent / "test"                       # {data_parent}/test
normal_folder = test_folder / "normal"                   # {data_parent}/test/normal
anomalous_folder = test_folder / "anomalous"             # {data_parent}/test/anomalous
```

**Usage**:
```python
# Old way (no longer works):
# model.test_folder("testing_data", output_folder="results")

# New way:
model.test_folder("trento_house_data", output_folder="results")
# Automatically uses: trento_house_data/test/normal and trento_house_data/test/anomalous
```

### 4. `test.py`
**Changes**: Updated `test_folder()` call to use new path

**Before**:
```python
my_model.test_folder("testing_data", output_folder="custom_test")
```

**After**:
```python
my_model.test_folder("trento_house_data", output_folder="custom_test")
```

### 5. `examples_refactored_model.py`
**Changes**: Updated all 7 example functions and comparison function

**Before**:
```python
model.fit("training_data")
results = model.test_folder("testing_data", output_folder="runs/example_1_results")
```

**After**:
```python
model.fit("trento_house_data/train")
results = model.test_folder("trento_house_data", output_folder="runs/example_1_results")
```

**Updated Functions**:
1. `example_1_simple_pca()` - ✅ Updated
2. `example_2_vae_gmm_mu()` - ✅ Updated
3. `example_3_vae_reconstruction()` - ✅ Updated
4. `example_4_high_sensitivity()` - ✅ Updated
5. `example_5_fast_robust()` - ✅ Updated
6. `example_6_single_prediction()` - ✅ Updated (also updated test image paths)
7. `example_7_compare_models()` - ✅ Updated
8. `example_8_logging()` - ✅ No changes needed

---

## Migration Guide

### For Existing Projects

If you have data in the old structure:
```
training_data/
testing_data/normal/
testing_data/anomalous/
```

**Option 1: Manual Migration**
```bash
# Create new structure
mkdir -p my_dataset/train
mkdir -p my_dataset/test/normal
mkdir -p my_dataset/test/anomalous

# Copy files
cp training_data/* my_dataset/train/
cp testing_data/normal/* my_dataset/test/normal/
cp testing_data/anomalous/* my_dataset/test/anomalous/
```

**Option 2: Use data_collect.py**
```bash
# Collect fresh data into new structure
python data_collect.py --mode train --out my_dataset
python data_collect.py --mode test --out my_dataset
```

### For New Projects

Simply use:
```bash
# Data collection automatically creates correct structure
python data_collect.py --mode train --out my_dataset
python data_collect.py --mode test --out my_dataset

# Training uses parent folder
python train.py  # Uses default "trento_house_data"
```

---

## Benefits of New Structure

✅ **Single Parent Path**: Pass one path instead of two (`train_folder` + `test_folder`)  
✅ **Cleaner API**: `test_folder(data_parent)` instead of `test_folder(test_folder_path)`  
✅ **Better Organization**: All related data in one folder hierarchy  
✅ **Easier Versioning**: Can have multiple datasets: `dataset_v1/`, `dataset_v2/`, etc.  
✅ **Reduced Duplication**: No need to pass paths separately throughout codebase  
✅ **Intuitive**: Follows common data organization patterns  

---

## Complete Workflow Example

```python
import torch
import open_clip
from train import TrainingOrchestrator, ConfigurationFactory, VisualEncoder
import logging

# Setup
device = "mps" if torch.backends.mps.is_available() else "cuda"
logger = logging.getLogger(__name__)

# Initialize orchestrator
orchestrator = TrainingOrchestrator(device, logger)

# Define configurations
configs = [
    ConfigurationFactory.create_balanced_pca_config(),
    ConfigurationFactory.create_vae_recon_kl_config(),
]

# Run training with single parent path
results = orchestrator.run(
    configs=configs,
    data_parent="my_dataset",  # All data under my_dataset/train and my_dataset/test
)

# Results automatically saved to training_runs/{timestamp}/
```

---

## Backward Compatibility Notes

⚠️ **Breaking Changes**:
- `test_folder()` now takes `data_parent` instead of `test_folder`
- `TrainingOrchestrator.run()` now takes `data_parent` instead of `train_folder`/`test_folder`
- `data_collect.py` now uses new `--out` default behavior

📝 **Migration Required**:
All scripts calling these methods must be updated to use new signatures.

---

## Troubleshooting

**Issue**: `FileNotFoundError: Expected /normal and /anomalous`  
**Solution**: Ensure your data structure matches:
```
data_parent/
├── train/          (for model.fit())
└── test/
    ├── normal/
    └── anomalous/
```

**Issue**: Old scripts still reference `training_data` or `testing_data`  
**Solution**: Update hardcoded paths to use new structure or use `data_collect.py` to generate.

**Issue**: "No images found in test folder"  
**Solution**: Verify test images are in `{data_parent}/test/normal/` or `{data_parent}/test/anomalous/`

---

## Files Modified Summary

| File | Changes | Status |
|------|---------|--------|
| `data_collect.py` | Updated `--out` flag behavior | ✅ |
| `train.py` | Updated `TrainingOrchestrator.run()` signature | ✅ |
| `models/model.py` | Updated `test_folder()` signature | ✅ |
| `test.py` | Updated test paths | ✅ |
| `examples_refactored_model.py` | Updated all 7 examples | ✅ |

---

## Next Steps

1. **Migrate Your Data**: Use Option 1 or 2 from migration guide above
2. **Update Custom Scripts**: Any scripts calling `test_folder()` or `run()`
3. **Test Integration**: Run a quick test with new structure:
   ```bash
   python data_collect.py --mode train --out test_data
   python train.py  # Verify it trains on test_data or modify script
   ```
4. **Deploy**: Use `data_parent` parameter for flexible dataset management

---

**Date**: October 30, 2025  
**Refactoring**: Directory structure optimization  
**Status**: ✅ Complete
