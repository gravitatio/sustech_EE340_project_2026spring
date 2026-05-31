import unittest
from unittest.mock import patch

from refuge_seg.train import build_model


class ModelConfigTests(unittest.TestCase):
    def test_rfau_cnxt_checkpoint_path_is_forwarded_to_model(self):
        config = {
            "model": {
                "name": "rfau_cnxt",
                "encoder_name": "convnext_large",
                "num_classes": 3,
                "pretrained": False,
                "checkpoint_path": "model.safetensors",
            }
        }

        with patch("refuge_seg.train.RFAUCNxtUNet") as model_cls:
            build_model(config)

        model_cls.assert_called_once_with(
            encoder_name="convnext_large",
            num_classes=3,
            pretrained=False,
            checkpoint_path="model.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
