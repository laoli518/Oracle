# ORACLE

Official code repository for the paper:

**ORACLE: Knowledge-Efficient Pig Behaviour Recognition via Ontology-Guided Contrastive Learning**

ORACLE is an ontology-guided vision-language framework for pig behaviour recognition. The model uses frozen CLIP image and text encoders with lightweight trainable projection heads, and supports evaluation with seen and unseen textual descriptions as well as few-shot downstream adaptation.

## Installation

```bash
git clone <repository-url>
cd oracle

conda create -n oracle python=3.10 -y
conda activate oracle

# Install PyTorch according to your CUDA environment first.
pip install -e .
```

Additional dependencies for downstream fine-tuning scripts using Excel annotations:

```bash
pip install pandas openpyxl scikit-learn opencv-python
```

## Dataset

The benchmark dataset contains image and video samples covering 24 pig behaviour categories.

**Dataset download:** [Link to be released / replace with the public dataset URL]

After downloading the dataset, prepare the JSON split files and media paths required by the training and evaluation scripts:

```text
data/
├── train-mp4.json
├── val.json
├── test.json
└── descriptions/
    └── test2_unseen_descriptions_example.json
```

## Running ORACLE

### 1. Train the base model

```bash
oracle-train \
  --train-data data/train-mp4.json \
  --val-data data/val.json \
  --output-dir outputs/training \
  --num-frames 25 \
  --motion-alpha 0.5 \
  --epochs 20 \
  --batch-size 128
```

The trained checkpoint is saved to:

```text
outputs/training/best_direct_contrastive_model.pth
```

### 2. Test on seen descriptions: `test1`

`test1` evaluates the trained model using the textual descriptions saved in the trained checkpoint.

```bash
oracle-test1 \
  --model-path outputs/training/best_direct_contrastive_model.pth \
  --test-data data/test.json \
  --output-dir outputs/test1_seen_descriptions
```

### 3. Test on unseen descriptions: `test2`

`test2` evaluates the trained model using external descriptions that were not used during training.

```bash
oracle-test2 \
  --model-path outputs/training/best_direct_contrastive_model.pth \
  --test-data data/test.json \
  --desc-file data/descriptions/test2_unseen_descriptions_example.json \
  --output-dir outputs/test2_unseen_descriptions
```

## Few-shot Fine-tuning

The `fine_tuning/` directory contains scripts for adapting a trained ORACLE checkpoint to three downstream pig production scenarios.

### Gestating-sow posture recognition

```bash
python fine_tuning/lying_posture_adapter_finetune_v11.py \
  --data_root /path/to/lying_images \
  --annotation /path/to/lying_annotations.xlsx \
  --model outputs/training/best_direct_contrastive_model.pth \
  --output_dir outputs/fine_tuning/lying_posture
```

### Feeding and drinking detection in growing-finishing pigs

```bash
python fine_tuning/drinking_eating_adapter_finetune_v11.py \
  --data_root /path/to/drinking_eating_images \
  --annotation /path/to/drinking_eating_annotations.xlsx \
  --model outputs/training/best_direct_contrastive_model.pth \
  --output_dir outputs/fine_tuning/drinking_eating
```

### Agonistic behaviour recognition in nursery pigs

```bash
python fine_tuning/fight_nofight_adapter_finetune_v12.py \
  --data_root /path/to/fight_media \
  --annotation /path/to/fight_annotations.xlsx \
  --model outputs/training/best_direct_contrastive_model.pth \
  --output_dir outputs/fine_tuning/fight_nofight \
  --n_frames 25
```

## Project Structure

```text
oracle/
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_training.py
├── run_test1.py
├── run_test2.py
├── data/
│   └── descriptions/
│       └── test2_unseen_descriptions_example.json
├── scripts/
│   ├── train_example.sh
│   ├── test1_example.sh
│   └── test2_example.sh
├── src/oracle/
│   ├── cli.py
│   ├── dataset.py
│   ├── features.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluation.py
│   └── external_evaluation.py
├── fine_tuning/
│   ├── fight_nofight_adapter_finetune_v12.py
│   ├── lying_posture_adapter_finetune_v11.py
│   └── drinking_eating_adapter_finetune_v11.py
├── tests/
└── legacy/
```

## Citation

Citation information will be added upon publication of the paper.

## Folder-based Fine-tuning Update

The current fine-tuning workflow reads datasets arranged as:

```text
dataset_root/
├── train/ClassA/
└── test/ClassA/
```

For example, Fight/No Fight fine-tuning can be run with:

```bash
python fine_tuning/finetune_fight_nofight.py \
  --dataset-root /path/to/baoyu \
  --model-path outputs/training/best_direct_contrastive_model.pth \
  --output-dir outputs/fine_tuning/baoyu_fight_nofight \
  --epochs 30 \
  --n-frames 25
```

See `fine_tuning/README.md` for the three downstream scenarios.
