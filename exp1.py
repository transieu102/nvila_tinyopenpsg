import argparse
import yaml
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import clip
from inference import PredicateClassificationResult
import os
from tqdm import tqdm

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp1.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)

def preprocess_response(response, max_words=20):
    words = response.split(" ")
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return response

def map_response_to_class(model, device, subject, response, obj, class_list, is_open_set=False):
    response = preprocess_response(response)
    query = response if is_open_set else f"{subject} {response} {obj}"
    candidates = [f"{subject} {c} {obj}" for c in class_list]
    with torch.no_grad():
        emb_query = normalize(model.encode_text(clip.tokenize([query]).to(device)))
        emb_cands = normalize(model.encode_text(clip.tokenize(candidates).to(device)))

    sims = torch.matmul(emb_cands, emb_query.T).squeeze()
    return class_list[torch.argmax(sims).item()]


def compute_metrics(y_true, y_pred, label_list):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=label_list, average="micro")
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, y_pred, labels=label_list, average="macro")
    print("Micro: P={:.3f}, R={:.3f}, F1={:.3f}".format(p, r, f1))
    print("Macro: P={:.3f}, R={:.3f}, F1={:.3f}".format(pm, rm, f1m))
    return classification_report(y_true, y_pred, labels=label_list, digits=3)


def per_class_f1(y_true, y_pred, label_list):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=label_list, average=None)
    return dict(zip(label_list, f1))


def plot_frequency_vs_f1(y_true, y_pred_zero, y_pred_finetune, all_classes, output_path):
    freq = Counter(y_true)
    f1_zero = per_class_f1(y_true, y_pred_zero, all_classes)
    f1_finetune = per_class_f1(y_true, y_pred_finetune, all_classes)

    sorted_labels = sorted(all_classes, key=lambda l: freq[l], reverse=True)
    counts = [freq[l] for l in sorted_labels]
    f1z = [f1_zero[l] for l in sorted_labels]
    f1f = [f1_finetune[l] for l in sorted_labels]

    x = np.arange(len(sorted_labels))
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.bar(x, np.array(counts) / len(y_true), color="lightgray", alpha=0.7, label="Class frequency (%)")
    ax1.set_ylabel("Class Frequency (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_labels, rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(x, f1z, marker="o", label="Zero-shot F1", color="tab:blue")
    ax2.plot(x, f1f, marker="o", label="Finetuned F1", color="tab:orange")
    ax2.set_ylabel("F1 Score")

    fig.legend(loc="upper right")
    plt.title("Class Frequency vs F1 Performance (Closed set, sorted by frequency)")
    plt.tight_layout()
    plt.savefig(output_path)


def plot_confusion(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(include_values=False, cmap="Blues", ax=ax, xticks_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)


def main():
    config = load_config()
    save_dir = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(config["clip_model"], device=device)

    # Load results
    with open(config["input_pickle"], "rb") as f:
        results: list[PredicateClassificationResult] = pickle.load(f)

    # Prepare data
    is_open_set = config["is_open_set"]
    if is_open_set:
        all_classes = sorted({r.gt_predicate for r in results if r.is_open_set})
        dataset = [r for r in results if r.is_open_set]
    else:
        all_classes = sorted({r.gt_predicate for r in results if not r.is_open_set})
        dataset = [r for r in results if not r.is_open_set]

    y_true, y_pred_zero, y_pred_finetune = [], [], []
    for r in tqdm(dataset, total=len(dataset)):
        y_true.append(r.gt_predicate)
        y_pred_zero.append(map_response_to_class(model, device, r.subject_class, r.pretrained_multi_choice, r.object_class, all_classes))
        y_pred_finetune.append(map_response_to_class(model, device, r.subject_class, r.finetuned_multi_choice, r.object_class, all_classes))

    # Metrics
    print("=== Zero-shot ===")
    report_zero = compute_metrics(y_true, y_pred_zero, all_classes)
    print(report_zero)

    print("=== Finetuned ===")
    report_finetune = compute_metrics(y_true, y_pred_finetune, all_classes)
    print(report_finetune)

    # Save reports to txt
    metrics_path = config["output"]["metrics_report"]
    with open(metrics_path, "w") as f:
        f.write("=== Zero-shot ===\n")
        f.write(report_zero + "\n")
        f.write("=== Finetuned ===\n")
        f.write(report_finetune + "\n")

    print(f"[INFO] Metrics report saved to {metrics_path}")


    # Visualization
    plot_frequency_vs_f1(y_true, y_pred_zero, y_pred_finetune, all_classes, config["output"]["freq_vs_f1"])
    plot_confusion(y_true, y_pred_zero, all_classes, "Zero-shot Confusion Matrix (Normalized)", config["output"]["confusion_zero"])
    plot_confusion(y_true, y_pred_finetune, all_classes, "Finetuned Confusion Matrix (Normalized)", config["output"]["confusion_finetune"])


if __name__ == "__main__":
    main()
