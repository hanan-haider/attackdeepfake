#%%writefile /kaggle/working/attackdeepfake/evaluate_models_correct.py
import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (auc, confusion_matrix,
                             precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from dfadetect import cnn_features
from dfadetect.agnostic_datasets.attack_agnostic_dataset import \
    AttackAgnosticDataset
from dfadetect.cnn_features import CNNFeaturesSetting
from dfadetect.models import models
from dfadetect.trainer import NNDataSetting
from dfadetect.utils import set_seed
from experiment_config import feature_kwargs

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    training_dataset_name: str,
    fake_dataset_name: str,
    path: str,
    lw: int = 2,
    save: bool = False,
) -> matplotlib.figure.Figure:
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save:
        fig.savefig(f"{path}.pdf")
    plt.close(fig)
    return fig


def calculate_eer(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute EER using bonafide=positive (pos_label=1) convention.
    y_true : ground-truth labels  (0=fake, 1=bonafide)
    y_score: sigmoid probabilities (higher → more bonafide)
    Returns thresh, eer, fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return thresh, eer, fpr, tpr, thresholds


def evaluate_nn(
    model_paths: List[Path],
    datasets_paths: List[Union[Path, str]],
    data_config: Dict,
    model_config: Dict,
    device: str,
    amount_to_use: Optional[int] = None,
    batch_size: int = 128,
):
    LOGGER.info("Loading data...")
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    use_cnn_features = False if model_name == "rawnet" else True
    cnn_features_setting = data_config.get("cnn_features_setting", None)

    nn_data_setting = NNDataSetting(use_cnn_features=use_cnn_features)
    cnn_features_setting = (
        CNNFeaturesSetting(**cnn_features_setting)
        if use_cnn_features else CNNFeaturesSetting()
    )

    weights_path = ''
    for fold in tqdm.tqdm(range(3)):
        model = models.get_model(
            model_name=model_name, config=model_parameters, device=device)
        if len(model_paths) > 1:
            assert len(model_paths) == 3, "Pass either 0 or 3 weights path"
            weights_path = model_paths[fold]
            model.load_state_dict(torch.load(weights_path))
        model = model.to(device)

        logging_prefix = f"fold_{fold}"
        data_val = AttackAgnosticDataset(
            #asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            #fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="val",
            reduced_number=amount_to_use,
        )
        LOGGER.info(
            f"Testing '{model_name}' model, weights: '{weights_path}', "
            f"on {len(data_val)} files."
        )
        print(f"\nTest Fold [{fold+1}/3]:")
        test_loader = DataLoader(
            data_val, batch_size=batch_size, drop_last=True, num_workers=3)

        num_correct = 0.0
        num_total   = 0.0
        y_pred       = torch.Tensor([]).to(device)   # sigmoid probabilities
        y            = torch.Tensor([]).to(device)   # ground-truth labels
        y_pred_label = torch.Tensor([]).to(device)   # hard binary predictions
        batches_number = len(data_val) // batch_size

        for i, (batch_x, _, batch_y) in enumerate(test_loader):
            model.eval()
            if i % 100 == 0:
                print(f"  Batch [{i}/{batches_number}]")
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                num_total += batch_x.size(0)
                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(
                        batch_x, cnn_features_setting=cnn_features_setting)
                batch_pred       = torch.sigmoid(model(batch_x).squeeze(1))
                batch_pred_label = (batch_pred + .5).int()
                num_correct     += (batch_pred_label == batch_y.int()).sum(dim=0).item()
                y_pred       = torch.concat([y_pred, batch_pred], dim=0)
                y_pred_label = torch.concat([y_pred_label, batch_pred_label], dim=0)
                y            = torch.concat([y, batch_y], dim=0)

        # ── Convert to numpy ──────────────────────────────────────────────────
        y_np       = y.cpu().numpy()               # float, 0=fake / 1=bonafide
        y_score_np = y_pred.cpu().numpy()          # sigmoid probs
        y_label_np = y_pred_label.cpu().numpy()    # hard 0/1 predictions
        y_int_np   = y_np.astype(int)
        y_label_int_np = y_label_np.astype(int)

        # ── Core metrics (all using probabilities, consistent convention) ─────
        eval_accuracy = (num_correct / num_total) * 100

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_int_np, y_label_int_np, average="binary", beta=1.0)

        # ✅ FIX 1: AUC uses soft probabilities, not hard binary labels
        auc_score = roc_auc_score(y_true=y_np, y_score=y_score_np)

        # ✅ FIX 2: EER uses same convention as ROC plot (bonafide=positive)
        thresh, eer, fpr, tpr, thresholds = calculate_eer(y_np, y_score_np)

        LOGGER.info(
            f"eval/{logging_prefix}__eer: {eer:.4f}, "
            f"eval/{logging_prefix}__accuracy: {eval_accuracy:.4f}, "
            f"eval/{logging_prefix}__precision: {precision:.4f}, "
            f"eval/{logging_prefix}__recall: {recall:.4f}, "
            f"eval/{logging_prefix}__f1_score: {f1_score:.4f}, "
            f"eval/{logging_prefix}__auc: {auc_score:.4f}"
        )
        print(
            f"  ✅ Fold {fold} done — "
            f"Accuracy: {eval_accuracy:.2f}%, "
            f"EER: {eer:.4f}, "
            f"AUC: {auc_score:.4f}"
        )

        # ── CONFUSION MATRIX ──────────────────────────────────────────────────
        cm = confusion_matrix(y_int_np, y_label_int_np)

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
        fig_cm.colorbar(im, ax=ax_cm)

        classes   = ['Fake (0)', 'Bonafide (1)']
        tick_marks = np.arange(len(classes))
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_xticklabels(classes, rotation=45, ha='right')
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_yticklabels(classes)

        thresh_cm = cm.max() / 2.0
        for ci in range(cm.shape[0]):
            for cj in range(cm.shape[1]):
                ax_cm.text(
                    cj, ci, format(cm[ci, cj], 'd'),
                    ha='center', va='center',
                    color='white' if cm[ci, cj] > thresh_cm else 'black'
                )

        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_title(f'Confusion Matrix — Fold {fold}')
        fig_cm.tight_layout()
        cm_path = f"/kaggle/working/confusion_matrix_fold_{fold}.png"
        plt.savefig(cm_path, dpi=150)
        plt.show()
        print(f"  📊 Confusion matrix saved → confusion_matrix_fold_{fold}.png")

        # ── ROC CURVE ─────────────────────────────────────────────────────────
        # ✅ FIX 3: Reuse the same fpr/tpr from calculate_eer (no 2nd roc_curve call)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--',
                    label='Random classifier')
        ax_roc.scatter(
            [eer], [1 - eer], color='red', zorder=5,
            label=f'EER point ({eer:.4f})'
        )
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curve — Fold {fold + 1}')
        ax_roc.legend(loc='lower right')
        fig_roc.tight_layout()
        roc_path = f"/kaggle/working/roc_curve_fold_{fold}.png"
        plt.savefig(roc_path, dpi=150)
        plt.show()
        print(f"  📈 ROC curve saved → roc_curve_fold_{fold}.png")


def evaluate_gmm(
    real_model_path: str,
    fake_model_path: str,
    datasets_paths: List[str],
    amount_to_use: Optional[int],
    feature_fn: Callable,
    feature_kwargs: dict,
    clusters: int,
    device: str,
    frontend: str,
    output_file_name: str,
    use_double_delta: bool = True
):
    complete_results = {}
    LOGGER.info(f"paths: {real_model_path}, {fake_model_path}, {datasets_paths}")

    for subtype in ["val", "test", "train"]:
        for fold in [0, 1, 2]:
            real_dataset_test = AttackAgnosticDataset(
                asvspoof_path=datasets_paths[0],
                wavefake_path=datasets_paths[1],
                fakeavceleb_path=datasets_paths[2],
                fold_num=fold, fold_subset=subtype,
                oversample=False, undersample=False,
                return_label=False, reduced_number=amount_to_use,
            )
            real_dataset_test.get_bonafide_only()

            fake_dataset_test = AttackAgnosticDataset(
                asvspoof_path=datasets_paths[0],
                wavefake_path=datasets_paths[1],
                fakeavceleb_path=datasets_paths[2],
                fold_num=fold, fold_subset=subtype,
                oversample=False, undersample=False,
                return_label=False, reduced_number=amount_to_use,
            )
            fake_dataset_test.get_spoof_only()

            real_dataset_test, fake_dataset_test = apply_feature_and_double_delta(
                [real_dataset_test, fake_dataset_test],
                feature_fn=feature_fn,
                feature_kwargs=feature_kwargs,
                use_double_delta=use_double_delta
            )

            model_path  = Path(real_model_path) / f"real_{fold}" / "ckpt.pth"
            real_model  = load_model(real_dataset_test, str(model_path), device, clusters)

            model_path  = Path(fake_model_path) / f"fake_{fold}" / "ckpt.pth"
            fake_model  = load_model(fake_dataset_test, str(model_path), device, clusters)

            plot_path = Path(f"plots/{frontend}/fold_{fold}/{subtype}")
            plot_path.mkdir(parents=True, exist_ok=True)

            results = {"fold": fold}
            LOGGER.info("Calculating on folds...")

            eer, thresh, fpr, tpr = calculate_eer_for_models(
                real_model, fake_model,
                real_dataset_test, fake_dataset_test,
                f"train_fold_{fold}", "all",
                plot_dir_path=str(plot_path), device=device,
            )
            results.update({
                "eer": str(eer), "thresh": str(thresh),
                "fpr": str(list(fpr)), "tpr": str(list(tpr))
            })
            LOGGER.info(f"{subtype} | Fold {fold}:\n\tEER: {eer} Thresh: {thresh}")
            complete_results.setdefault(subtype, {})[fold] = results

    with open(f"{output_file_name}.json", "w+") as f:
        json.dump(complete_results, f, indent=4)


def main(args):
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["data"].get("seed", 42))

    if not args.use_gmm:
        evaluate_nn(
            model_paths=config["checkpoint"].get("paths", []),
            datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
            model_config=config["model"],
            data_config=config["data"],
            amount_to_use=args.amount,
            device=device,
        )
    else:
        evaluate_gmm(
            real_model_path=args.ckpt,
            fake_model_path=args.ckpt,
            datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
            feature_fn=lfcc if args.lfcc else mfcc,
            feature_kwargs=feature_kwargs(lfcc=args.lfcc),
            clusters=args.clusters,
            device=device,
            frontend="lfcc" if args.lfcc else "mfcc",
            amount_to_use=args.amount,
            output_file_name="gmm_evaluation",
            use_double_delta=True
        )


def parse_args():
    parser = argparse.ArgumentParser()

    ASVSPOOF_DATASET_PATH   = "../datasets/ASVspoof2021/LA"
    WAVEFAKE_DATASET_PATH   = "../datasets/WaveFake"
    FAKEAVCELEB_DATASET_PATH = "../datasets/FakeAVCeleb/FakeAVCeleb_v1.2"

    parser.add_argument("--asv_path",      type=str, default=ASVSPOOF_DATASET_PATH)
    parser.add_argument("--wavefake_path", type=str, default=WAVEFAKE_DATASET_PATH)
    parser.add_argument("--celeb_path",    type=str, default=FAKEAVCELEB_DATASET_PATH)
    parser.add_argument("--config",  type=str,  default="config.yaml")
    parser.add_argument("--amount", "-a", type=int, default=None)
    parser.add_argument("--cpu",    "-c", action="store_true")
    parser.add_argument("--use_gmm",      action="store_true")
    parser.add_argument("--clusters", "-k", type=int, default=128)
    parser.add_argument("--lfcc",    "-l", action="store_true")
    parser.add_argument("--output",  "-o", type=str, default="results")
    parser.add_argument("--ckpt",          type=str, default="trained_models")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())