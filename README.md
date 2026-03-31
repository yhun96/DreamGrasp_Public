# DreamGrasp_Public

Public implementation and examples for running DreamGrasp with demo data or custom data.

---

## Installation

First, clone this repository and `threestudio`, then place the custom DreamGrasp module under `threestudio/custom`.

```bash
git clone https://github.com/yhun96/DreamGrasp_Public
git clone https://github.com/threestudio-project/threestudio # tested on threestudio v0.1.0
mv DreamGrasp threestudio/custom
````

---

## Environment Setup

The following setup was tested with CUDA 12.4, Python 3.10.14 / 3.11, and PyTorch 2.6.0 + cu124.

1. Install a CUDA driver compatible with your GPU.

   * Tested on CUDA 12.4

2. Install Python.

   * Tested on Python 3.10.14 and 3.11

```bash
conda create -n dreamgrasp python=3.10.14
conda activate dreamgrasp
```

3. Install a PyTorch version compatible with your GPU.

   * Tested on PyTorch 2.6.0 + cu124

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

4. Install the remaining Python dependencies.

```bash
pip install -r requirements.txt
```

5. Install `simple-knn`.

```bash
cd threestudio/custom
git clone https://github.com/DSaurus/simple-knn.git
pip install ./simple-knn
```

---

## Download Zero123

Download the Zero123 checkpoint `stable-zero123.ckpt` from https://huggingface.co/stabilityai/stable-zero123 and place it in `threestudio/load/zero123`.

---

## Run the Demo

```bash
cd threestudio
python launch.py --config custom/DreamGrasp/configs/dreamgrasp.yaml --train --gpu 0
```

The outputs are saved in:

```bash
threestudio/outputs/dreamgrasp/exp
```

---

## Download Datasets

The demo dataset can be downloaded from Google Drive:

https://drive.google.com/file/d/1xQahIXKU9reHsjp-Ogx-TDf9J_kNP46P/view?usp=sharing

Please place the extracted files so that the path becomes `DreamGrasp_Public/datasets/Objects_2/...`.

---

# Using Custom Data

To run DreamGrasp with custom data, you need:

* one config `.yaml` file
* `rgba_0.png`, ..., `rgba_k.png`
* `mask_0.npy`, ..., `mask_k.npy`
* `prompts.txt`

where `k >= 2`.

An example config file is provided at: `DreamGrasp/configs/dreamgrasp.yaml`

After preparing the data according to the guidelines below and modifying the required values in the config file, run:

```bash
cd threestudio
python launch.py --config {your_config_path}.yaml --train --gpu 0
```

For preprocessing and postprocessing examples, please refer to the notebooks in the `notebook` folder.

---

## Camera pose convention

Objects are assumed to be placed near the origin frame `(0, 0, 0)`.

The camera frames should satisfy the following assumptions:

* each camera should face toward the origin / workspace center,
* all camera centers should lie on the surface of a sphere with the same radius.

Please refer to `notebooks/raw_Data_visualization.ipynb` for an example.

Under this assumption, you can set the following fields in the config file:

```yaml
data:
    default_elevation_deg:
    default_azimuth_deg:
```

These values should match the actual elevation and azimuth of your camera views.

For camera distance, do **not** modify the config file.
Instead, we compensate for the difference by rescaling the output mesh afterward.
Please refer to `notebooks/mesh_postprocess.ipynb`.
