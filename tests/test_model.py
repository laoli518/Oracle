import torch
from oracle.model import CustomCLIPContrastiveTrainer


def test_projection_shapes_and_normalization():
    torch.manual_seed(42)
    model = CustomCLIPContrastiveTrainer(feature_dim=512)
    image = torch.randn(4, 512)
    text = torch.randn(4, 512)
    out_image, out_text = model(image, text)
    assert out_image.shape == (4, 512)
    assert out_text.shape == (4, 512)
    assert torch.allclose(out_image.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(out_text.norm(dim=-1), torch.ones(4), atol=1e-5)
