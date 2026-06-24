import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union


import torch
import yaml


from dfadetect.agnostic_datasets.attack_agnostic_dataset import AttackAgnosticDataset
from dfadetect.cnn_features import CNNFeaturesSetting
from dfadetect.models import models
from dfadetect.trainer import GDTrainer, NNDataSetting
from dfadetect.utils import set_seed
from experiment_config import feature_kwargs


LOGGER = logging.getLogger()


def init_logger(log_file):
    LOGGER.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)


def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")


def train_nn(
    datasets_paths: List[Union[Path, str]],
    batch_size: int,
    epochs: int,
    device: str,
    model_config: Dict,
    cnn_features_setting: CNNFeaturesSetting,
    oversample: bool = False,
    undersample: bool = False,
    model_dir: Optional[Path] = None,
    amount_to_use: Optional[int] = None,
) -> None:

    LOGGER.info("Loading data...")
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    optimizer_config = model_config["optimizer"]

    use_cnn_features = False if model_name == "rawnet" else True

    nn_data_setting = NNDataSetting(
        use_cnn_features=use_cnn_features,
    )
    timestamp = time.time()
    folds_number = 3

    for fold in range(folds_number):

        data_train = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="train",
            reduced_number=amount_to_use,
            oversample=oversample,
            undersample=undersample,
        )

        data_val = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="test",
            reduced_number=amount_to_use,
            oversample=oversample,
        )

        data_test = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="val",
            reduced_number=amount_to_use,
        )

        # ── Dataset Statistics ────────────────────────────────────────────
        train_label_counts = data_train.samples["label"].value_counts()
        val_label_counts   = data_test.samples["label"].value_counts()
        test_label_counts  = data_val.samples["label"].value_counts()

        train_bonafide = train_label_counts.get("bonafide", 0)
        train_spoof    = train_label_counts.get("spoof",    0)
        val_bonafide   = val_label_counts.get("bonafide",   0)
        val_spoof      = val_label_counts.get("spoof",      0)
        test_bonafide  = test_label_counts.get("bonafide",  0)
        test_spoof     = test_label_counts.get("spoof",     0)

        LOGGER.info(
            f"\n{'='*70}\n"
            f"  Fold {fold} — Dataset Statistics\n"
            f"{'='*70}\n"
            f"  TRAIN            → total: {train_bonafide + train_spoof:>7,} "
            f"| bonafide: {train_bonafide:>6,} | spoof: {train_spoof:>7,}\n"
            f"  Validation (test) → total: {val_bonafide   + val_spoof:>7,} "
            f"| bonafide: {val_bonafide:>6,} | spoof: {val_spoof:>7,}\n"
            f"  Test     (val)  → total: {test_bonafide  + test_spoof:>7,} "
            f"| bonafide: {test_bonafide:>6,} | spoof: {test_spoof:>7,}\n"
            f"  Spoof ratio  — train: {train_spoof / max(train_bonafide, 1):.2f}:1"
            f"  | val: {val_spoof / max(val_bonafide, 1):.2f}:1"
            f"  | test: {test_spoof / max(test_bonafide, 1):.2f}:1\n"
            f"{'='*70}"
        )

        # ── pos_weight for BCEWithLogitsLoss (train split only) ───────────
        pos_weight_value = train_spoof / max(train_bonafide, 1)
        LOGGER.info(f"  pos_weight for fold {fold}: {pos_weight_value:.3f}")


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.verbose:
        LOGGER.setLevel(logging.INFO)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    set_seed(seed)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not args.use_gmm:
        cnn_features_setting = config["data"].get("cnn_features_setting", None)
        if cnn_features_setting:
            cnn_features_setting = CNNFeaturesSetting(**cnn_features_setting)
        else:
            cnn_features_setting = CNNFeaturesSetting()

        train_nn(
            datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
            device=device,
            amount_to_use=args.amount,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_dir=model_dir,
            model_config=config["model"],
            cnn_features_setting=cnn_features_setting,
            oversample=args.oversample,
            undersample=args.undersample,
        )
    else:
        feature_fn = lfcc if args.lfcc else mfcc
        train_gmm(
            datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
            feature_fn=feature_fn,
            feature_kwargs=feature_kwargs(args.lfcc),
            clusters=args.clusters,
            batch_size=args.batch_size,
            device=device,
            model_dir=model_dir,
            use_double_delta=True,
            amount_to_use=args.amount
        )


def parse_args():
    parser = argparse.ArgumentParser()

    ASVSPOOF_DATASET_PATH    = "../datasets/ASVspoof2021/LA"
    WAVEFAKE_DATASET_PATH    = "../datasets/WaveFake"
    FAKEAVCELEB_DATASET_PATH = "../datasets/FakeAVCeleb/FakeAVCeleb_v1.2"

    parser.add_argument("--asv_path",      type=str, default=ASVSPOOF_DATASET_PATH)
    parser.add_argument("--wavefake_path", type=str, default=WAVEFAKE_DATASET_PATH)
    parser.add_argument("--celeb_path",    type=str, default=FAKEAVCELEB_DATASET_PATH)

    parser.add_argument("--config",  type=str, default="config.yaml")
    parser.add_argument("--amount",  "-a", type=int,  default=None)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--epochs",  "-e", type=int,  default=5)
    parser.add_argument("--ckpt",    type=str, default="trained_models")
    parser.add_argument("--cpu",     "-c", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    parser.add_argument(
        "--oversample",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        "--undersample",
        action=argparse.BooleanOptionalAction,
        default=False
    )

    parser.add_argument("--use_gmm",   action="store_true")
    parser.add_argument("--clusters",  "-k", type=int, default=128)
    parser.add_argument("--lfcc",      "-l", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())