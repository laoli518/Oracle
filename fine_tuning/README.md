# ORACLE Fine-tuning
## Dataset format

Example for the nursery-pig agonistic behaviour task:

```text
nursery/
├── train/
│   ├── Fight/
│   └── No Fight/
└── test/
    ├── Fight/
    └── No Fight/
```

Folder names are matched in a case/space/underscore-insensitive way. For example, `No Fight`, `no_fight`, and `no-fight` can all match the class `No Fight`.

## Fight / No Fight

```bash
python fine_tuning/finetune_fight_nofight.py \
  --dataset-root /path/to/nursery \
  --model-path outputs/training/best_direct_contrastive_model.pth \
  --output-dir outputs/fine_tuning/nursery_fight_nofight \
```

## Lying posture

Expected folders:

```text
lying_posture/train/Lateral Lying
lying_posture/train/Sternal Lying
lying_posture/train/Not Lying
lying_posture/test/Lateral Lying
lying_posture/test/Sternal Lying
lying_posture/test/Not Lying
```

Run:

```bash
python fine_tuning/finetune_lying_posture.py \
  --dataset-root /path/to/lying_posture \
  --model-path outputs/training/best_direct_contrastive_model.pth \
  --output-dir outputs/fine_tuning/lying_posture \
  --epochs 30
```

## Drinking / Eating

Expected folders:

```text
drinking_eating/train/Drinking
drinking_eating/train/Eating
drinking_eating/train/Other
drinking_eating/test/Drinking
drinking_eating/test/Eating
drinking_eating/test/Other
```

Run:

```bash
python fine_tuning/finetune_drinking_eating.py \
  --dataset-root /path/to/drinking_eating \
  --model-path outputs/training/best_direct_contrastive_model.pth \
  --output-dir outputs/fine_tuning/drinking_eating \
  --epochs 30
```

## Outputs

```text
output_dir/
├── best_adapter_model.pth
├── fine_tuning_results.json
├── train_predictions.csv
└── test_predictions.csv
```

## Notes

- The trained ORACLE `.pth` checkpoint is loaded as the base model.
- CLIP encoders and ORACLE projection heads are frozen.
- Only image and text residual adapters are trained.
- Images and videos are both supported. Videos are sampled using `--n-frames`.
