# CauSight: Learning to Supersense for Visual Causal Discovery

## 🔧 User Guide

### 1. Clone the Repository

```bash
git clone https://github.com/OpenCausaLab/CauSight.git
cd CauSight
```

### 2. Set Up the Environment

We recommend using **conda**:

```bash
conda create -n causight python=3.10
conda activate causight

pip install -r requirements.txt
pip install -e .
```

### 3. Download the Dataset (VCG-32K)

```bash
mkdir -p VCG-32K
pip install huggingface_hub

hf login
hf download OpenCausaLab/VCG-32K \
    --repo-type dataset \
    --local-dir ./VCG-32K
```

```bash
tar -xzf ./VCG-32K/COCO/images.tar.gz -C ./VCG-32K/COCO
tar -xzf ./VCG-32K/365/images.tar.gz -C ./VCG-32K/365
```

### 4. Download the CauSight Model

```bash
mkdir -p model
huggingface-cli download OpenCausaLab/CauSight \
    --repo-type model \
    --local-dir ./model
```

### 5. Evaluation

Start the model server, then run inference:

```bash
bash model_server.sh
python run_inference.py
```

### 6. Tree-of-Causal-Thought (If you want to make your own SFT data with ToCT.)

```bash
bash model_server.sh
python run.py
```
