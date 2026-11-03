"""Extra figures for the Claude-free classifier deck. Reuses the standard ones from make_figures.py (copied into the same folder)."""
import json
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casefilter.rules import classify_rules  # noqa: E402

BLUE, RED, GRAY, LBLUE, LRED, GREEN = "#0F4C81", "#C8423B", "#8A8F98", "#8DB4DC", "#E8A39F", "#4C9A6A"
plt.rcParams.update({"font.family": ["Microsoft YaHei", "Arial"], "axes.unicode_minus": False, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555"})
src_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for f in ("f2_gates", "f3_dataset", "f6_leakage", "f7_cv_C", "f8_roc_pr", "f9_threshold", "f10_confusion_metrics", "f11_prob_hist", "f12_pca"):
    shutil.copy(src_dir / f"{f}.png", out / f"{f}.png")
R = Path("out")


def save(fig, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


def box(ax, x, y, w, h, text, fc="white", ec=BLUE, tc="black", fs=12, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=BLUE, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=color, lw=1.8, ls=ls))


# ---- N1 pipeline (inference path without any LLM) + offline training labels ---------------
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")
ax.text(0.1, 5.3, "推理 (每篇文献)：本地运行，无需联网和 API", fontsize=13, fontweight="bold", color=BLUE)
box(ax, 0.1, 3.7, 2.0, 1.2, "全文文件\nxml / txt / pdf", fc="#EAF1F8", bold=True)
box(ax, 2.7, 3.7, 2.1, 1.2, "① 文本提取\n取最完整的一份")
box(ax, 5.4, 3.7, 2.5, 1.2, "② 切块 + PubMedBERT\n逐块向量化")
box(ax, 8.5, 3.7, 2.4, 1.2, "③ 池化\n全文均值 ⊕ 首段均值")
box(ax, 5.4, 1.9, 5.5, 1.2, "④ 逻辑回归分类头 → P(入库)\nP ≥ 0.5 判为 1 (入库), 否则判为 0", fc=BLUE, ec=BLUE, tc="white", bold=True)
box(ax, 0.1, 1.9, 4.5, 1.2, "输出：概率 + 标签 0 / 1\n(casefilter predict → CSV)", fc="#EAF1F8", bold=True)
for a in [(2.1, 4.3, 2.7, 4.3), (4.8, 4.3, 5.4, 4.3), (7.9, 4.3, 8.5, 4.3)]: arrow(ax, *a)
arrow(ax, 9.7, 3.7, 8.3, 3.1); arrow(ax, 5.4, 2.5, 4.6, 2.5)
ax.text(0.1, 1.2, "训练 (离线, 只做一次)：用已标注文献拟合分类头", fontsize=13, fontweight="bold", color=RED)
box(ax, 0.1, 0.1, 5.0, 0.8, "标签来源：精选文献 90 篇 + Claude 离线标注 884 篇", fc="#FBEDEC", ec=RED, fs=11.5)
box(ax, 5.7, 0.1, 5.2, 0.8, "权重 weights/head_v3.npz (1536 系数 + 截距)", fc="white", ec=RED, fs=11.5)
arrow(ax, 5.1, 0.5, 5.7, 0.5, RED)
save(fig, "n1_pipeline")

# ---- N2 chunking + pooling ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, 0.1, 1.8, 1.9, 1.2, "全文\n(~2.6 万字符)", fc="#EAF1F8", bold=True)
for i in range(6):
    face = "#CFE0F1" if i < 2 else "white"
    box(ax, 2.6 + i * 0.62, 2.6 - (i % 2) * 0.15, 0.52, 0.52, "", fc=face, ec=BLUE)
    ax.text(2.86 + i * 0.62, 2.86 - (i % 2) * 0.15, str(i + 1), ha="center", va="center", fontsize=10)
ax.text(4.45, 3.5, "切成 480 token 的块 (最多 12 块)\n前 2 块 (标题/摘要) 一定保留", ha="center", fontsize=11, linespacing=1.4)
box(ax, 2.6, 0.5, 3.7, 0.9, "PubMedBERT 句向量 (每块 768 维)", fc="white")
box(ax, 7.0, 2.7, 3.9, 0.9, "全文块均值 (768 维)", fc="#EAF1F8")
box(ax, 7.0, 1.4, 3.9, 0.9, "前 2 块均值 (768 维)", fc="#EAF1F8")
box(ax, 7.0, 0.1, 3.9, 0.9, "拼接 → 1536 维文献特征", fc=BLUE, ec=BLUE, tc="white", bold=True)
arrow(ax, 2.0, 2.4, 2.6, 2.7); arrow(ax, 4.45, 2.5, 4.45, 1.4); arrow(ax, 6.3, 1.0, 7.0, 3.1); arrow(ax, 6.3, 0.9, 7.0, 1.8)
ax.text(8.95, 2.3, "⊕", ha="center", va="center", fontsize=20, color=RED, fontweight="bold"); arrow(ax, 8.95, 1.4, 8.95, 1.0)
save(fig, "n2_embedding")

# ---- training-set reconstruction (same logic and seed as retrain_v3.py) ------------------
df = pd.read_pickle(R / "predictions.pkl").reset_index(drop=True)
fl = pd.read_csv(R / "final_labels.csv", dtype={"pmid": str}, encoding="utf-8-sig").set_index("pmid")
cl = fl[fl.label_source.str.startswith("claude")]
df["claude"] = df.pmid.map(cl.label)
df["y"] = np.where(df.claude.notna(), df.claude, np.where(df.gold & df.claude.isna(), 1, -1))
lab = df.y >= 0
cidx = df.index[df.claude.notna()]
strat = df.loc[cidx, "y"].astype(int).astype(str) + df.loc[cidx, "source"]
_, va_idx = train_test_split(cidx, test_size=0.3, random_state=0, stratify=strat)
tr = np.array(sorted(set(df.index[lab]) - set(va_idx)))
va = np.array(sorted(va_idx))
dtr = df.loc[tr]

# ---- N3 source-balanced weights ----------------------------------------------------------
cnt = dtr.groupby(["source", dtr.y.astype(int)]).size().unstack(fill_value=0).reindex(["xml", "txt", "pdf"])
# same rule as retrain_v3.weights(): 1/cell-size per sample, each class rescaled to total 0.5, then divided by the mean weight
raw = 1.0 / cnt.astype(float)
yy, ss = dtr.y.astype(int).values, dtr.source.values
wv = np.array([raw.loc[s_, l_] for s_, l_ in zip(ss, yy)])
for l_ in (0, 1):
    wv[yy == l_] *= 0.5 / wv[yy == l_].sum()
wv = wv / wv.mean()
rel = pd.DataFrame({l_: [wv[(ss == s_) & (yy == l_)].mean() for s_ in ("xml", "txt", "pdf")] for l_ in (0, 1)}, index=["xml", "txt", "pdf"])
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
for ax, M, title, fmt in ((axs[0], cnt, "训练集篇数", "{:d}"), (axs[1], rel, "每篇样本的相对权重", "{:.2f}")):
    ax.imshow(M.values.astype(float), cmap="Blues", aspect="auto")
    for (i, j), v in np.ndenumerate(M.values):
        ax.text(j, i, fmt.format(int(v) if fmt == "{:d}" else v), ha="center", va="center", fontsize=15, fontweight="bold",
                color="white" if M.values[i, j] > M.values.max() * 0.6 else "black")
    ax.set_xticks([0, 1], ["标签 0", "标签 1"]); ax.set_yticks(range(3), ["XML", "TXT", "PDF"]); ax.set_title(title, fontweight="bold")
    for s_ in ax.spines.values(): s_.set_visible(False)
fig.suptitle("来源均衡: 每个 (来源, 标签) 格子的总权重相同", fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout(); save(fig, "n3_weights")

# ---- N4 train / validation split ---------------------------------------------------------
dva = df.loc[va]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
parts = [("训练", (dtr.y == 1).sum(), (dtr.y == 0).sum()), ("验证 (留出)", (dva.y == 1).sum(), (dva.y == 0).sum())]
for i, (n_, p_, q_) in enumerate(parts):
    ax.barh(i, p_, color=BLUE, label="标签 1" if i == 0 else None); ax.barh(i, q_, left=p_, color=RED, label="标签 0" if i == 0 else None)
    ax.text(p_ / 2, i, str(p_), ha="center", va="center", color="white", fontsize=13)
    ax.text(p_ + q_ / 2, i, str(q_), ha="center", va="center", color="white", fontsize=13)
ax.set_yticks([0, 1], [f"{p[0]}\n(n={p[1] + p[2]})" for p in parts]); ax.invert_yaxis(); ax.set_xlabel("篇数")
ax.legend(frameon=False, loc="lower right"); ax.set_title("训练集与验证集 (按 标签×来源 分层, 70/30)", fontweight="bold")
save(fig, "n4_split")

# ---- N5 rules-only vs embedding head vs all-ones on the validation set -------------------
corpus = {}
for line in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(line); corpus[r["pmid"]] = r
yv = dva.y.astype(int).values
v3 = pd.read_csv(R / "predictions_v3.csv", dtype={"pmid": str}, encoding="utf-8-sig").set_index("pmid")
pv = np.array([v3.prob_v3.get(p, np.nan) for p in dva.pmid])
ok = ~np.isnan(pv)
rl = np.array([classify_rules(dict(pmid=p, title=corpus[p]["title"], abstract="", fulltext=corpus[p]["text"])).label for p in dva.pmid])


def m(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return [precision_score(y, pred, zero_division=0), recall_score(y, pred), tn / (tn + fp), f1_score(y, pred), balanced_accuracy_score(y, pred)]


res = {"规则 (INCLUDE 判 1)": m(yv, (rl == "INCLUDE").astype(int)), "规则 (INCLUDE+REVIEW 判 1)": m(yv, (rl != "EXCLUDE").astype(int)),
       "嵌入 + 分类头": m(yv[ok], (pv[ok] >= 0.5).astype(int)), "全判 1": m(yv, np.ones_like(yv))}
labs = ["精确率", "召回率", "特异度", "F1", "平衡准确率"]
fig, ax = plt.subplots(figsize=(11, 4.6)); x = np.arange(5); w_ = 0.2
for k, ((n_, v_), c_) in enumerate(zip(res.items(), (LRED, GRAY, BLUE, LBLUE))):
    b = ax.bar(x + (k - 1.5) * w_, v_, w_, color=c_, label=n_); ax.bar_label(b, fmt="%.2f", fontsize=8.5)
ax.set_xticks(x, labs); ax.set_ylim(0, 1.42); ax.legend(frameon=False, ncol=2, loc="upper center")
ax.set_title(f"验证集 (n={int(ok.sum())}) 上不同无大模型方案的比较", fontweight="bold")
save(fig, "n5_compare")
json.dump({k: [round(float(x), 3) for x in v] for k, v in res.items()}, open(out / "n5_numbers.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: [round(float(x), 3) for x in v] for k, v in res.items()}, ensure_ascii=False))
print("train", len(dtr), "val", len(dva), "cnt\n", cnt, "\nrel\n", rel.round(2))
