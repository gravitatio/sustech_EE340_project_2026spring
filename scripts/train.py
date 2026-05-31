import argparse

from refuge_seg.config import load_config
from refuge_seg.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train REFUGE optic disc/cup segmentation model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Only load config and print the planned run.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print("Dry run config loaded:")
        print(config)
        return
    train_from_config(config)


if __name__ == "__main__":
    main()

