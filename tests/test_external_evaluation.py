import json
from pathlib import Path

import torch

from oracle.external_evaluation import (
    DescriptionManager,
    evaluate_pos_neg_margin_accuracy,
    normalize_label,
)
from oracle.model import CustomCLIPContrastiveTrainer


def test_description_manager_reads_zero_shot_fields(tmp_path: Path):
    path = tmp_path / "descriptions.json"
    path.write_text(json.dumps({"Eating": {"positive_zs": ["eat"], "negative_zs": ["sleep"]}}))
    manager = DescriptionManager.from_file(str(path))
    assert normalize_label("  Lateral   Lying ") == "lateral lying"
    assert manager.resolve(" eating ") == "Eating"
    assert manager.get_positive("EATING") == ["eat"]
    assert manager.get_negative("eating") == ["sleep"]


def test_pos_neg_margin_accuracy_uses_prompt_margin():
    manager = DescriptionManager({"Eating": {"positive_zs": ["pos"], "negative_zs": ["neg"]}})
    model = CustomCLIPContrastiveTrainer(feature_dim=2)
    model.image_projection = torch.nn.Identity()
    model.text_projection = torch.nn.Identity()
    media_features = {"a.jpg": torch.tensor([1.0, 0.0]), "b.jpg": torch.tensor([-1.0, 0.0])}
    text_features = {"pos": torch.tensor([1.0, 0.0]), "neg": torch.tensor([-1.0, 0.0])}
    label_samples = {"Eating": [{"media_path": "a.jpg"}, {"media_path": "b.jpg"}]}
    result = evaluate_pos_neg_margin_accuracy(
        model, label_samples, manager, media_features, text_features, torch.device("cpu")
    )
    assert result["overall_accuracy"] == 50.0
    assert result["per_class_detail"]["Eating"] == {"correct": 1, "total": 2}
    assert len(result["errors"]) == 1
