import os
import argparse
import yaml
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from PIL import Image
import clip

from inference import PredicateClassificationResult


# ================= HELPERS =================
def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def preprocess_response(response, max_words=20):
    """Clean response text to avoid tokens like [1,2,3]."""
    import re
    cleaned = re.sub(r"\[\d+(?:,\d+)*\]", "", response)
    words = cleaned.split(" ")
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return cleaned.strip()


def embed_texts(model, device, texts, batch_size=64):
    """Return L2-normalized embeddings (numpy)."""
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = clip.tokenize(batch).to(device)
            embs = model.encode_text(tokens)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


def threshold_recall(similarities, thresholds=np.linspace(0, 1, 50)):
    recalls = []
    total = len(similarities)
    for t in thresholds:
        tp = np.sum(similarities >= t)
        fn = total - tp
        recalls.append(tp / (tp + fn))
    return thresholds, np.array(recalls), np.mean(recalls)


def per_class_recall(similarities, gt_classes, thresholds, best_thr):
    """Compute per-class recall at best threshold."""
    recalls = {}
    class_set = sorted(set(gt_classes))
    for c in class_set:
        idx = [i for i, g in enumerate(gt_classes) if g == c]
        sims = similarities[idx]
        tp = np.sum(sims >= best_thr)
        fn = len(idx) - tp
        recalls[c] = tp / (tp + fn) if len(idx) > 0 else 0.0
    cmr = np.mean(list(recalls.values()))
    return recalls, cmr


def plot_threshold_recall(thr, rec_zs, ar_zs, rec_ft, ar_ft, out_path):
    plt.figure(figsize=(8, 6))
    plt.plot(thr, rec_zs, label=f"Zero-shot (AR={ar_zs:.3f})")
    plt.plot(thr, rec_ft, label=f"Finetune (AR={ar_ft:.3f})")
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Threshold–Recall Curve (Triplet Detection)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_per_class_recall(recalls_zs, recalls_ft, freq, out_path):
    labels = sorted(freq.keys(), key=lambda l: freq[l], reverse=True)
    rzs = [recalls_zs.get(l, 0) for l in labels]
    rft = [recalls_ft.get(l, 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, rzs, width, label="Zero-shot")
    plt.bar(x + width/2, rft, width, label="Finetune")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Recall")
    plt.title("Per-class Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def show_samples(meta, diffs, title, out_path, n_show=10):
    """
    Show top-n samples with largest differences.

    Args:
        meta: list of dicts, each dict contains sample info including 'image', 'gt_triplet', etc.
        diffs: numpy array or list, chênh lệch score cho mỗi sample
        title: str, title for the figure
        out_path: str, path to save figure
        n_show: int, số sample muốn show
    """
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return

    # Chọn top-n indices với chênh lệch lớn nhất
    top_idxs = np.argsort(-diffs)[:n_show]  # lấy top n theo giá trị tuyệt đối

    n = len(top_idxs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    
    for i, idx in enumerate(top_idxs):
        m = meta[idx]
        plt.subplot(rows, cols, i + 1)
        if isinstance(m["image"], Image.Image):
            plt.imshow(m["image"])
        else:
            try:
                plt.imshow(np.asarray(m["image"]))
            except:
                plt.text(0.5, 0.5, "No image", ha="center")
        plt.axis("off")
        s = f"GT: {m.get('gt_triplet', '')}\nZS: {preprocess_response(m.get('pretrained_basic', ''))}\nFT: {preprocess_response(m.get('finetuned_basic', ''))}\nDiff: {diffs[idx]:.3f}"
        plt.title(s, fontsize=7)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def plot_diversity(zs_embs, ft_embs, out_dir):
    # pairwise similarity distribution
    def pairwise(embs):
        sims = cosine_similarity(embs)
        iu = np.triu_indices(sims.shape[0], k=1)
        return sims[iu]

    pair_zs = pairwise(zs_embs)
    pair_ft = pairwise(ft_embs)

    plt.figure(figsize=(8, 4))
    sns.kdeplot(pair_zs, label="Zero-shot", fill=True)
    sns.kdeplot(pair_ft, label="Finetune", fill=True)
    plt.xlabel("Pairwise cosine similarity")
    plt.title("Distribution of pairwise similarities")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "pairwise_sim.png"))
    plt.close()

    # PCA scatter
    combined = np.vstack([zs_embs, ft_embs])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)
    n = zs_embs.shape[0]
    proj_zs, proj_ft = proj[:n], proj[n:]

    plt.figure(figsize=(8, 6))
    plt.scatter(proj_zs[:, 0], proj_zs[:, 1], marker="o", label="Zero-shot", alpha=0.6, edgecolor="k", s=40)
    plt.scatter(proj_ft[:, 0], proj_ft[:, 1], marker="^", label="Finetune", alpha=0.6, edgecolor="k", s=40)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("PCA of embeddings (circle=ZS, triangle=FT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca.png"))
    plt.close()


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp2.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("output_dir", "exp2_outputs")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(cfg.get("clip_model", "ViT-B/32"), device=device)

    # Load data
    results = load_results(cfg["input_pickle"])
    subset = cfg.get("subset", "open")
    if subset == "open":
        results = [r for r in results if r.is_open_set]
    elif subset == "closed":
        results = [r for r in results if not r.is_open_set]
    print(f"Loaded {len(results)} samples")

    # Prepare texts
    gt_texts, zs_texts, ft_texts, meta, gt_classes = [], [], [], [], []
    for i, r in enumerate(results):
        gt = r.gt_triplet
        zs = preprocess_response(r.pretrained_basic)
        ft = preprocess_response(r.finetuned_basic)

        gt_texts.append(gt)
        zs_texts.append(zs)
        ft_texts.append(ft)
        gt_classes.append(r.gt_predicate)
        # y_true.append(1 if r.is_open_set else 0)
        meta.append({
            "image": r.image,
            "gt_triplet": r.gt_triplet,
            "gt_predicate": r.gt_predicate,
            "pretrained_basic": r.pretrained_basic,
            "finetuned_basic": r.finetuned_basic,
            "is_open_set": r.is_open_set
        })

    # Embed
    print("Embedding texts...")
    gt_embs = embed_texts(model, device, gt_texts)
    zs_embs = embed_texts(model, device, zs_texts)
    ft_embs = embed_texts(model, device, ft_texts)

    # Similarities
    sim_zs = np.sum(gt_embs * zs_embs, axis=1)
    sim_ft = np.sum(gt_embs * ft_embs, axis=1)

    # Threshold–Recall
    thr = np.linspace(0, 1, 50)
    thr, rec_zs, ar_zs = threshold_recall(sim_zs, thr)
    _, rec_ft, ar_ft = threshold_recall(sim_ft, thr)
    plot_threshold_recall(thr, rec_zs, ar_zs, rec_ft, ar_ft,
                          os.path.join(out_dir, "threshold_recall.png"))

    # zs_correct_ft_wrong = np.where((sim_zs > sim_ft))[0]
    # ft_correct_zs_wrong = np.where((sim_ft > sim_zs))[0]
    # print(f"ZS better than FT: {len(zs_correct_ft_wrong)}")
    # print(f"FT better than ZS: {len(ft_correct_zs_wrong)}")
    zs_better_than_ft = sim_zs - sim_ft
    ft_better_than_zs = sim_ft - sim_zs

    show_samples(meta, zs_better_than_ft,
                 "ZS better than FT",
                 os.path.join(out_dir, "zs_better_than_ft.png"))
    show_samples(meta, ft_better_than_zs,
                 "FT better than ZS",
                 os.path.join(out_dir, "ft_better_than_zs.png"))

    # Diversity
    plot_diversity(zs_embs, ft_embs, out_dir)

   

    print("Experiment 2 finished. Results saved in", out_dir)


if __name__ == "__main__":
    main()
