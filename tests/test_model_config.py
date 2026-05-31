import unittest
from unittest.mock import patch

from refuge_seg.models.rfau_cnxt import remap_convnext_feature_keys
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


class ConvNeXtCheckpointKeyTests(unittest.TestCase):
    def test_remaps_original_convnext_keys_to_timm_feature_list_keys(self):
        state_dict = {
            "stem.0.weight": "stem-conv",
            "stem.1.bias": "stem-norm",
            "stages.0.blocks.0.gamma": "stage0",
            "stages.3.downsample.0.weight": "stage3",
            "head.fc.weight": "classifier",
        }

        remapped = remap_convnext_feature_keys(state_dict)

        self.assertEqual(remapped["stem_0.weight"], "stem-conv")
        self.assertEqual(remapped["stem_1.bias"], "stem-norm")
        self.assertEqual(remapped["stages_0.blocks.0.gamma"], "stage0")
        self.assertEqual(remapped["stages_3.downsample.0.weight"], "stage3")
        self.assertNotIn("head.fc.weight", remapped)
