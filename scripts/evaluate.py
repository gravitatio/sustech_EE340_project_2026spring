import argparse

from refuge_seg.config import load_config
from refuge_seg.evaluate import predict_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict REFUGE masks from a trained checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()
    predict_from_checkpoint(load_config(args.config), args.checkpoint, split=args.split, postprocess=not args.no_postprocess)


if __name__ == "__main__":
    main()
