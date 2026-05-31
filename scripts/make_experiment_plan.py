from pathlib import Path


CONFIGS = [
    "configs/rfau_cnxt_large_ce.yaml",
    "configs/rfau_cnxt_large_dice.yaml",
    "configs/rfau_cnxt_large_ce_dice.yaml",
    "configs/rfau_cnxt_large_ce_dice_lr3e4_bs16.yaml",
    "configs/segformer_b5_ce_dice.yaml",
]


def main() -> None:
    print("# REFUGE experiment commands")
    for cfg in CONFIGS:
        name = Path(cfg).stem
        print(f"python scripts/train.py --config {cfg}")
        print(f"python scripts/evaluate.py --config {cfg} --checkpoint outputs/{name}/best.pt --split val")


if __name__ == "__main__":
    main()
