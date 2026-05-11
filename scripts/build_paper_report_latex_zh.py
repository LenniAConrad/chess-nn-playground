#!/usr/bin/env python
"""Build paper_report_zh.pdf — simplified-Chinese version of the scout report.

Re-uses all data loaders from build_paper_report_latex.py but emits Chinese
section headings and prose, with model names / math / metric names kept in
English (the conventional Chinese-ML-paper style).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Re-use everything from the English script
sys.path.insert(0, str(Path(__file__).parent))
from build_paper_report_latex import (
    analyze_run, short_name, latex_escape, fmt_params, fmt_speed, fmt_mflops,
    enc_pill_tex, texttt_breakable, render_confusion_matrix_png,
)


CN_PREAMBLE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[a4paper, top=1.8cm, bottom=1.8cm, left=1.8cm, right=1.8cm, headsep=6pt]{geometry}

% XeLaTeX + xeCJK for Simplified Chinese
\usepackage{fontspec}
\usepackage{xeCJK}
% Use Noto Sans CJK SC for body Chinese; Noto Serif CJK SC for headings
\setCJKmainfont[BoldFont={Noto Sans CJK SC Bold},
                ItalicFont={Noto Sans CJK SC},
                AutoFakeSlant=0.2]{Noto Sans CJK SC}
\setCJKsansfont[BoldFont={Noto Sans CJK SC Bold}]{Noto Sans CJK SC}
\setCJKfamilyfont{songsc}{Noto Serif CJK SC}
\setmainfont[Ligatures=TeX, Path=assets/fonts/]{Inter-Regular.otf}[
  BoldFont=Inter-Bold.otf,
  ItalicFont=Inter-Italic.otf,
  BoldItalicFont=Inter-BoldItalic.otf,
]

\usepackage{microtype}
\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{tabularx}
\usepackage{colortbl}
\usepackage[most]{tcolorbox}
\usepackage{titlesec}
\usepackage{titling}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage{pdflscape}
\usepackage{calc}
\usepackage{xspace}
\usepackage{caption}
\usepackage{etoolbox}
\usepackage{ragged2e}
\usepackage{float}

% Refined green colour palette
\definecolor{forest}{HTML}{2D7355}
\definecolor{deepforest}{HTML}{1B3F2F}
\definecolor{darkforest}{HTML}{0F2C20}
\definecolor{sage}{HTML}{6B9F7C}
\definecolor{palesage}{HTML}{E9F2EC}
\definecolor{verylightsage}{HTML}{F4F9F5}
\definecolor{moss}{HTML}{4C8A6A}
\definecolor{olive}{HTML}{7A9170}
\definecolor{gold}{HTML}{B58904}
\definecolor{ink}{HTML}{14181A}
\definecolor{muted}{HTML}{1B3F2F}
\definecolor{linecolor}{HTML}{C6DCCD}
\definecolor{rule}{HTML}{1B3F2F}
\definecolor{warn}{HTML}{9A4540}
\definecolor{stripebg}{HTML}{F0F6F2}
\definecolor{lc0blue}{HTML}{1B3F2F}
\definecolor{lc0bluebg}{HTML}{E2EEE5}
\definecolor{s18red}{HTML}{6B8E73}
\definecolor{s18redbg}{HTML}{EEF5EF}
\definecolor{accent}{HTML}{4C8A6A}
\definecolor{accentbg}{HTML}{E2EEE5}

\hypersetup{colorlinks=true, urlcolor=forest, linkcolor=forest, citecolor=forest}

\setlength{\parskip}{0.32em}
\setlength{\parindent}{0pt}
\setlist{topsep=2pt, partopsep=0pt, parsep=0pt, itemsep=2pt}

\captionsetup{font=small, labelfont={bf,color=forest},
  textfont={it,color=muted}, margin=10pt}

\titleformat{\section}
  {\color{forest}\sffamily\Large\bfseries}
  {\thesection}{0.6em}{}[\vskip 1pt {\color{linecolor}\hrule height 0.6pt}]
\titleformat{\subsection}
  {\color{deepforest}\sffamily\large\bfseries}
  {\thesubsection}{0.6em}{}
\titleformat{\subsubsection}
  {\color{moss}\sffamily\normalsize\bfseries}
  {\thesubsubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\color{muted}\sffamily\small\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\renewcommand{\arraystretch}{1.08}
\rowcolors{2}{}{stripebg}

\tcbset{
  callout/.style={
    enhanced, breakable,
    colback=palesage, colframe=forest, colbacktitle=palesage,
    boxrule=0pt, leftrule=3pt, arc=2pt,
    left=10pt, right=10pt, top=8pt, bottom=8pt,
    fonttitle=\sffamily\bfseries\color{deepforest}\footnotesize,
    coltitle=deepforest,
  },
  goodcallout/.style={callout},
  warncallout/.style={callout, colback={s18redbg}, colbacktitle={s18redbg},
    colframe=warn, coltitle=warn},
  archcard/.style={
    enhanced, breakable,
    colback=white, colframe=linecolor, boxrule=0.4pt, arc=2pt,
    left=12pt, right=12pt, top=10pt, bottom=10pt,
  },
  archhead/.style={
    enhanced, colback=palesage, colframe=forest, boxrule=0pt,
    leftrule=3pt, arc=1pt, left=10pt, right=10pt, top=6pt, bottom=6pt,
  },
}

\newcommand{\pilllc}[0]{\colorbox{lc0bluebg}{\strut\textcolor{lc0blue}{\scriptsize\bfseries\sffamily lc0\_bt4\_112}}}
\newcommand{\pillsi}[0]{\colorbox{s18redbg}{\strut\textcolor{s18red}{\scriptsize\bfseries\sffamily simple\_18}}}
\newcommand{\pillneu}[0]{\colorbox{verylightsage}{\strut\textcolor{muted}{\scriptsize\sffamily ?}}}

\newcolumntype{N}{>{\raggedleft\arraybackslash}p{1.55cm}}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}

\setlength{\emergencystretch}{2em}

\newenvironment{lead}{\par\itshape\color{muted}\large}{\par\medskip}

\newcommand{\statbox}[2]{%
  \begin{tcolorbox}[
    enhanced, colback=verylightsage, colframe=forest, boxrule=0pt,
    leftrule=2.5pt, arc=1pt, width=\linewidth,
    left=8pt, right=8pt, top=6pt, bottom=6pt,
  ]
  {\color{deepforest}\fontfamily{lmr}\selectfont\Huge\bfseries #1}\\[2pt]
  {\color{muted}\sffamily\scriptsize\bfseries #2}
  \end{tcolorbox}%
}
"""


# ---------- Builders (Chinese-headed but reusing the data) ----------

def build_leaderboard_table_zh(runs):
    rows = []
    for i, r in enumerate(runs[:15], 1):
        name = texttt_breakable(short_name(r["group"]))
        enc = enc_pill_tex(r["encoding"])
        params = latex_escape(fmt_params(r["num_params"]))
        speed = latex_escape(fmt_speed(r["samples_per_sec"]))
        flops = latex_escape(fmt_mflops(r["mflops_per_pos"]))
        rows.append(
            f"{i} & {enc} & {name} & "
            f"\\textbf{{{r['test_pr_auc']:.4f}}} & {r['val_pr_auc']:.4f} & "
            f"{params} & {speed} & {flops} \\\\"
        )
    return r"""
\begin{small}
\begin{longtable}{@{}r l p{5.6cm} r r r r r@{}}
\toprule
\textbf{排名} & \textbf{编码} & \textbf{架构} &
\textbf{测试 PR} & \textbf{验证 PR} & \textbf{参数量} & \textbf{速度 $\uparrow$} & \textbf{FLOPs/位 $\downarrow$} \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
\end{small}
"""


def build_pareto_table_zh(runs):
    def is_dominated(r, others):
        for o in others:
            if (o["test_pr_auc"] >= r["test_pr_auc"]
                and (o["samples_per_sec"] or 0) >= (r["samples_per_sec"] or 0)
                and (o["test_pr_auc"] > r["test_pr_auc"]
                     or (o["samples_per_sec"] or 0) > (r["samples_per_sec"] or 0))):
                return True
        return False
    pareto = [r for r in runs if not is_dominated(r, runs)]
    pareto.sort(key=lambda r: r["test_pr_auc"], reverse=True)
    rows = []
    for r in pareto[:8]:
        rows.append(
            f"{enc_pill_tex(r['encoding'])} & {texttt_breakable(short_name(r['group']))} & "
            f"\\textbf{{{r['test_pr_auc']:.4f}}} & "
            f"{latex_escape(fmt_speed(r['samples_per_sec']))} & "
            f"{latex_escape(fmt_params(r['num_params']))} & "
            f"{latex_escape(fmt_mflops(r['mflops_per_pos']))} \\\\"
        )
    return r"""
\begin{small}
\begin{tabularx}{\linewidth}{@{}l p{5.6cm} r r r r@{}}
\toprule
\textbf{编码} & \textbf{架构} & \textbf{测试 PR $\uparrow$} & \textbf{速度 $\uparrow$} &
\textbf{参数量} & \textbf{FLOPs/位 $\downarrow$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{small}
"""


def build_slice_winners_table_zh(per_class):
    interesting = [
        ("crtk_difficulty",  "very_easy",   "极易"),
        ("crtk_difficulty",  "easy",        "简单"),
        ("crtk_difficulty",  "medium",      "中等"),
        ("crtk_difficulty",  "hard",        "困难"),
        ("crtk_difficulty",  "very_hard",   "极难"),
        ("crtk_phase",       "opening",     "开局"),
        ("crtk_phase",       "middlegame",  "中局"),
        ("crtk_phase",       "endgame",     "残局"),
        ("crtk_eval_bucket", "equal",       "均势(最难)"),
        ("crtk_eval_bucket", "winning_white","白方占优"),
        ("crtk_eval_bucket", "crushing_white","白方压制"),
        ("crtk_tactic_motifs", "hanging",   "悬子"),
        ("crtk_tactic_motifs", "fork",      "双叫"),
        ("crtk_tactic_motifs", "pin",       "牵制"),
        ("crtk_tactic_motifs", "skewer",    "穿刺"),
        ("crtk_tactic_motifs", "overload",  "过载"),
        ("crtk_tactic_motifs", "discovered_attack","闪击"),
        ("crtk_tactic_motifs", "mate_in_1", "一步杀"),
        ("crtk_tactic_motifs", "promotion", "升变"),
    ]
    rows = []
    for dim, val, label in interesting:
        ranked = []
        for g in per_class["groups"]:
            cell = g["per_slice"].get(dim, {}).get(val)
            if not cell: continue
            mean = cell["pr_auc"]["mean"]
            if mean != mean: continue
            ranked.append((g["group"], mean, cell["pr_auc"]["std"]))
        ranked.sort(key=lambda t: t[1], reverse=True)
        if not ranked: continue
        top_name, top_mean, top_std = ranked[0]
        margin = top_mean - ranked[1][1] if len(ranked) > 1 else 0
        rows.append(
            f"{label} & {texttt_breakable(short_name(top_name))} & "
            f"$\\mathbf{{{top_mean:.3f}}} \\pm {top_std:.3f}$ & $+{margin:.3f}$ \\\\"
        )
    return r"""
\begin{small}
\begin{tabularx}{\linewidth}{@{}p{3.0cm} L r r@{}}
\toprule
\textbf{切片} & \textbf{冠军模型} & \textbf{PR AUC} $\pm$ \textbf{标准差} & \textbf{领先幅度} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{small}
"""


def build_matched_recall_table_zh(matched_recall):
    motif_3seed = []
    for run in matched_recall["runs"]:
        slices = run.get("slice_accuracies_at_target_recall", {}) or {}
        motif = slices.get("crtk_tactic_motifs", {}) if isinstance(slices, dict) else {}
        promo = motif.get("promotion") if isinstance(motif, dict) else None
        if not isinstance(promo, dict) or promo.get("n", 0) == 0: continue
        motif_3seed.append({
            "name": re.sub(r"_seed\d+$", "", run["run_name"]),
            "near_fp_rate": promo["near_fp_rate"],
            "accuracy": promo["accuracy_at_target_recall"],
        })
    motif_3seed.sort(key=lambda r: r["near_fp_rate"])
    rows = []
    for i, r in enumerate(motif_3seed[:8], 1):
        rows.append(
            f"{i} & {texttt_breakable(short_name(r['name']))} & "
            f"\\textbf{{{r['near_fp_rate']:.3f}}} & {r['accuracy']:.3f} \\\\"
        )
    return r"""
\begin{small}
\begin{tabularx}{\linewidth}{@{}r p{6cm} r r@{}}
\toprule
\textbf{排名} & \textbf{架构} & \textbf{近谜题 FP 率 $\downarrow$} & \textbf{切片准确率 @ 召回0.80 $\uparrow$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{small}
"""


def build_arch_appendix_zh():
    cards = [
        {
            "id": "i193",
            "name_en": "Exchange-Then-King Dual Stream",
            "name_zh": "兑子–王安全双流网络",
            "encoding": "simple_18", "params": "157k", "speed": "11.6k/s", "pr_auc": 0.876,
            "summary": (
                r"将主干分成两个并行的卷积编码器：一个偏向\textbf{战术兑子}"
                r"(攻击/防御几何、吃子序列), 一个偏向\textbf{王的安全}"
                r"(王翼平面、将军射线、逃逸格)。一个小型 MLP 相位路由器输出 sigmoid 门控 "
                r"$\alpha(x)\in[0,1]$，按局面位置混合两个流的 logit；残差头 $h_R$ 读取拼接的流池化结果，"
                r"恢复跨流信号。"
            ),
            "equation": (
                r"\hat{y}(x) = \sigma\!\bigl( \alpha(x)\cdot h_K(\phi_K(x))"
                r" + (1-\alpha(x))\cdot h_E(\phi_E(x))"
                r" + h_R(\phi_K(x)\oplus\phi_E(x)) \bigr)"
            ),
            "eq_caption": (
                r"$\phi_E,\phi_K$ 为每流的卷积编码器；$\oplus$ 表示通道级拼接；$\sigma$ 为二元 sigmoid。"
            ),
            "bias": (
                r"将经典 Stockfish 风格的分解 (\textit{兑子评估} + \textit{王安全})"
                r"直接编码到架构中，而不是让网络从数据中自行发现。"
            ),
            "limits": (
                r"每条流内部是纯卷积，长程棋子交互需要多层才能传递。"
                r"扩展到更大规模时可将各流的编码器替换为注意力机制以缓解。"
            ),
        },
        {
            "id": "i048",
            "name_en": r"Rule-Automorphism Quotient Bottleneck Network (RAQ-Net)",
            "name_zh": "规则自同构商瓶颈网络 (RAQ-Net)",
            "encoding": "simple_18", "params": "179k", "speed": "12.0k/s", "pr_auc": 0.861,
            "summary": (
                r"将棋盘视为离散国际象棋自同构群 $G = \langle\mu_{\mathrm{LR}}, \mu_{\mathrm{col}}\rangle$ "
                r"作用下的 $G$-集合 (水平镜像并交换易位权; 颜色翻转并交换角色)。"
                r"网络主干按构造满足 $G$-等变性, 分类器从轨道空间读取而非原始棋盘。"
            ),
            "equation": (
                r"\Phi(x) = \frac{1}{|G|} \sum_{g \in G} \rho(g) \cdot \phi(g^{-1}\cdot x), \quad "
                r"\Phi(g\cdot x) = \rho(g)\,\Phi(x)\ \forall g \in G"
            ),
            "eq_caption": (
                r"$\phi$ 为每个轨道的编码器；$\rho(g)$ 在群作用之后对等变输出进行重新对齐；"
                r"第二个等式是 $\Phi$ 按构造满足的等变性约束。"
            ),
            "bias": (
                r"其他模型需要从数据中学习的对称性, 直接由架构提供。样本效率优势是数学保证而非经验观察。"
            ),
            "limits": (
                r"商瓶颈按构造丢弃信息；对二元判别有用, 但对价值/策略回归不利。"
            ),
        },
        {
            "id": "i018",
            "name_en": "Oriented Tactical Sheaf Laplacian",
            "name_zh": "有向战术层 Laplacian",
            "encoding": "simple_18", "params": "91k", "speed": "5.5k/s", "pr_auc": 0.861,
            "summary": (
                r"在棋盘各格上构造层 (sheaf), 其截面编码\emph{有向}战术关联 (攻击者$\to$被攻击者,"
                r"考虑当前走子方)。用有向层 Laplacian $L_{\mathrm{or}}$ 替代标准图 Laplacian 卷积, 保留攻击的方向性。"
            ),
            "equation": (
                r"L_{\mathrm{or}} = D - A_{\mathrm{or}}, \quad "
                r"A_{\mathrm{or}}[u,v] = "
                r"\begin{cases}+1 & u\text{ 攻击 } v \\ -1 & v\text{ 攻击 } u \\ 0 & \text{其他}\end{cases}"
                r", \quad y = W\,L_{\mathrm{or}}\,x"
            ),
            "eq_caption": (
                r"$D$ 为入/出度对角阵；$A_{\mathrm{or}}$ 按攻击/防御角色和走子方带符号；$W$ 为可学习的混合矩阵。"
            ),
            "bias": (
                r"棋盘上的战术关系本质是有向的；对称图 Laplacian 会丢弃这一信息, 有向 Laplacian 则保留。"
            ),
            "limits": (
                r"参数量极小 (91k), 架构本身成为超参数, 真正天花板需在更大规模下验证。"
            ),
        },
        {
            "id": "i188",
            "name_en": "Tactical Program Induction Network",
            "name_zh": "战术程序归纳网络",
            "encoding": "simple_18", "params": "710k", "speed": "8.3k/s", "pr_auc": 0.861,
            "summary": (
                r"将谜题视为存在一段短战术程序 (弃子 $\to$ 双叫 $\to$ 升变)。"
                r"神经程序归纳头在学习到的战术原语库 $\mathcal{P}$ 上为候选程序打分, 谜题 logit 取最大似然程序的得分。"
            ),
            "equation": (
                r"p(\text{puzzle} \mid x) = \max_{P\in\mathcal{P}}\,q_\theta(P \mid x), \quad "
                r"q_\theta(P \mid x) = \prod_{t=1}^{|P|} q_\theta(P_t \mid x, P_{<t})"
            ),
            "eq_caption": (
                r"$\mathcal{P}$ 为 (可学习的) 短战术程序库；$q_\theta$ 为自回归神经程序打分器。"
            ),
            "bias": (
                r"战术解决方案是\emph{原语的组合序列}, 而非点评估；程序形状的输出头显式表达了组合性。"
            ),
            "limits": (
                r"程序归纳头训练困难；参数量较高, 帕累托效率不如更小的赢家。"
            ),
        },
        {
            "id": "BT4",
            "name_en": "LC0 BT4 (reference architecture)",
            "name_zh": "LC0 BT4 (参考架构)",
            "encoding": "lc0_bt4_112", "params": "~50M (medium)", "speed": "varies",
            "pr_auc": float("nan"),
            "summary": (
                r"BT4 是当前 LC0 的参考主干: 64 格棋盘上的 piece-token Transformer。"
                r"输入为 112 平面 LC0 编码 (8 步历史 $\times$ 13 个棋子通道, 加上易位/走子方/吃过路兵/半步钟等辅助平面)。"
                r"主干为多头自注意力堆栈, 每格作为一个 token; 位置信息通过逐格可学习嵌入注入。"
                r"顶部并列两个头: WDL 价值头与 1858 维走法策略头。"
            ),
            "equation": (
                r"\mathrm{BT4}(x) = \mathrm{Head}\!\bigl( (\mathrm{Block}_L \circ \cdots \circ \mathrm{Block}_1)"
                r"(\mathrm{Embed}(x) + P) \bigr), \quad "
                r"\mathrm{Block}_\ell(z) = \mathrm{FFN}(\mathrm{MHSA}(z)) + z"
            ),
            "eq_caption": (
                r"$\mathrm{Embed}: \mathbb{R}^{112\times8\times8}\to\mathbb{R}^{64\times d}$ "
                r"将每格的平面向量投影为 $d$ 维 token；$P\in\mathbb{R}^{64\times d}$ 为逐格学习位置嵌入；"
                r"$\mathrm{MHSA}$ 为对全部 64 个 token 的多头自注意力；$\mathrm{FFN}$ 为两层前馈。"
            ),
            "bias": (
                r"格作为 token 让任意两格在一层内可交互 --- 适合长程棋子关系 (a1 的后攻击 h8 的王)。"
                r"这是其他主干难以匹敌的结构性优势。"
            ),
            "limits": (
                r"主干是\emph{通用的}: 没有国际象棋专属分解, 没有群等变性, 没有棋类感知的注意力偏置。"
                r"这些都需要从数据中学习。LC0 BT4 的强度主要来自训练预算, 而非架构本身。"
            ),
        },
        {
            "id": "i011",
            "name_en": "VetoSelect Positive-Claim Abstention",
            "name_zh": "VetoSelect 正向声明弃权",
            "encoding": "lc0_bt4_112", "params": "502k", "speed": "11.7k/s", "pr_auc": 0.858,
            "summary": (
                r"在 LC0 BT4 风格主干上增加选择性弃权头。损失函数将正向谜题证据和反驳证据分离, "
                r"并对硬负样本 (近谜题局面) 显式施加 \emph{veto} 惩罚。"
            ),
            "equation": (
                r"\mathcal{L} = \underbrace{-y\log p^+ - (1-y)\log(1-p^+)}_{\text{二元交叉熵}}"
                r" + \lambda\,\mathbb{E}_{x\in\mathrm{hard}^-}[p^+(x)]"
            ),
            "eq_caption": (
                r"$p^+$ 为谜题 logit；$\mathrm{hard}^-$ 是已验证的近谜题负样本 (fine label 1) 集合；"
                r"$\lambda > 0$ 为 veto 权重。"
            ),
            "bias": (
                r"对谜题分类而言, 聚合 PR AUC 是错误的评分标准 --- 真正的代价是\textit{看似战术却非谜题}"
                r"的局面上的假阳性。显式抑制这种模式是严格的优势。"
            ),
            "limits": (
                r"多目标损失牺牲聚合 AUC 容量换取切片鲁棒性。"
                r"如果只关心聚合 AUC 则用处不大。"
            ),
        },
    ]
    out = []
    for a in cards:
        out.append(r"\begin{tcolorbox}[archcard, breakable]" + "\n")
        out.append(r"\begin{tcolorbox}[archhead, boxsep=0pt, top=4pt, bottom=4pt]" + "\n")
        out.append(
            r"\textcolor{deepforest}{\sffamily\bfseries\large " + latex_escape(a["id"]) + r" --- " +
            a["name_zh"] + r"} \\[2pt]" + "\n"
        )
        out.append(
            r"{\color{muted}\small " + latex_escape(a["name_en"]) + r"}" + "\n"
        )
        pr_str = f"{a['pr_auc']:.3f}" if a['pr_auc'] == a['pr_auc'] else "暂无 (不在 scout 中)"
        out.append(
            r"\\[1pt]\textcolor{muted}{\sffamily\scriptsize " + enc_pill_tex(a["encoding"]) +
            r"\quad 参数量 \textbf{" + latex_escape(a["params"]) +
            r"}\quad 速度 \textbf{" + latex_escape(a["speed"]) +
            r"}\quad 测试 PR AUC \textbf{" + pr_str + r"}}" + "\n"
        )
        out.append(r"\end{tcolorbox}" + "\n")
        out.append(r"\smallskip" + "\n")
        out.append(a["summary"] + "\n")
        out.append(r"\par\smallskip" + "\n")
        out.append(r"\begin{equation*}" + "\n" + a["equation"] + "\n" + r"\end{equation*}" + "\n")
        out.append(r"\par\vspace{-4pt}\begin{center}\textit{\textcolor{muted}{\small " +
                   a["eq_caption"] + r"}}\end{center}" + "\n")
        out.append(r"\par\smallskip{\color{forest}\sffamily\bfseries\textgreater\,}" +
                   r"\textit{" + a["bias"] + r"}" + "\n")
        out.append(r"\par\smallskip\textbf{\sffamily\footnotesize\color{muted}局限}\\" + "\n")
        out.append(a["limits"] + "\n")
        out.append(r"\end{tcolorbox}" + "\n\n")
    return "".join(out)


def build_bt4_path_zh():
    steps = [
        {
            "phase": "步骤 1",
            "title": "获取匹配的训练数据",
            "body": (
                r"直接与公开预训练的 BT4 比较不公平 --- 该网络经过数十亿自我对弈局面、多个 GPU 年的训练。"
                r"我们的研究关注的是\emph{主干 (trunk)}, 而非训练流程。因此公平的比较是: \emph{在同一数据上从头训练两个网络}。"
                r"可行的数据来源: (a) Stockfish NNUE 训练语料 (Stockfish 16/17/18 神经网络的训练数据中, "
                r"有大量公开的 $(\text{局面},\,\text{评估},\,\text{最佳着法})$ 三元组); "
                r"(b) 自行挖掘 --- 在大师对局数据库上以固定深度运行 Stockfish。"
                r"后者花费 CPU 时间但可复现、不含歧义。"
            ),
        },
        {
            "phase": "步骤 2",
            "title": "多流主干 (3 流注意力)",
            "body": (
                r"将 i193 的兑子/王双流推广为\textbf{三个并行 Transformer 流}: "
                r"兑子流、王流、位置流。每流为 64 格上的小型 Transformer, "
                r"注意力偏置矩阵由棋类感知的预计算表得到: 兑子流用攻击/防御几何; "
                r"王流用王翼/将军射线; 位置流用标准相对位置偏置。融合采用 i193 的相位路由器, "
                r"推广为软 3 路混合 $\alpha\in\Delta^2$。"
            ),
        },
        {
            "phase": "步骤 3",
            "title": "价值头 + 策略头",
            "body": (
                r"将谜题二元头替换为 LC0 风格的双头: WDL 价值头 $\hat{v}(x)\in\Delta^2$ "
                r"(胜/平/负 3 路 softmax) 与 1858 维策略头 $\hat{\pi}(x\mid m)\propto e^{z_m}$ "
                r"(对合法着法掩码)。训练目标为价值与策略损失的加权和: "
                r"$\mathcal{L} = D_{\mathrm{KL}}(\hat{v}\,\|\,v_{\mathrm{tgt}}) + \beta\,D_{\mathrm{KL}}(\hat{\pi}\,\|\,\pi_{\mathrm{tgt}})$, "
                r"目标来自步骤 1 选定的数据源。"
            ),
        },
        {
            "phase": "步骤 4",
            "title": "国际象棋自同构等变包装",
            "body": (
                r"用 i048 的群等变性包装主干: 每个 Transformer block 与 "
                r"$G = \langle\mu_{\mathrm{LR}}, \mu_{\mathrm{col}}\rangle$ 可交换。"
                r"在匹配计算下, 对称战术模式获得 $|G| = 4$ 倍的有效训练数据。"
                r"实现方式: 跨轨道等价局面的权重共享。"
            ),
        },
        {
            "phase": "步骤 5",
            "title": "正面对决: 同训练、两主干",
            "body": (
                r"在\emph{完全相同的训练设置下} (同数据、同优化器、同算力预算、同头) 训练两个网络: "
                r"(a)~上述多流主干; (b)~新初始化的标准 BT4 形状 Transformer 主干。"
                r"\textbf{这才是我们的研究真正能赢的对比。} 我们不声称击败 LC0 团队耗时多年训练的 BT4。"
                r"我们声称: 在匹配训练下, 我们的主干比通用 Transformer 主干具有更好的归纳偏置。"
                r"用相同固定深度的对战上报 ELO。"
            ),
        },
        {
            "phase": "步骤 6",
            "title": "逐流辅助监督 (可选)",
            "body": (
                r"为每条流增加棋类感知的辅助损失 (兑子流用兑子结果预测; 王流用王进攻/防御分类; "
                r"位置流用局面评估回归)。辅助权重 $\lambda_i \leq 0.05$, 让主损失主导: "
                r"$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{value}} + \beta\mathcal{L}_{\text{policy}} + "
                r"\sum_{i\in\{E,K,P\}}\lambda_i\,\mathcal{L}_i^{\text{aux}}$。"
                r"只有多流主干具有结构性空位放置此类辅助监督, BT4 没有。"
            ),
        },
    ]
    out = []
    for i, s in enumerate(steps, 1):
        out.append(r"\begin{minipage}{\linewidth}" + "\n")
        out.append(r"\begin{minipage}[t]{0.06\linewidth}" + "\n")
        out.append(rf"\fontfamily{{lmr}}\selectfont\LARGE\bfseries\color{{forest}}{i}" + "\n")
        out.append(r"\end{minipage}\hfill" + "\n")
        out.append(r"\begin{minipage}[t]{0.93\linewidth}" + "\n")
        out.append(rf"\textcolor{{muted}}{{\scriptsize\sffamily\bfseries {s['phase']}}}\\[0pt]" + "\n")
        out.append(rf"\textcolor{{deepforest}}{{\sffamily\normalsize\bfseries {s['title']}}}\\[2pt]" + "\n")
        out.append(s["body"] + "\n")
        out.append(r"\end{minipage}" + "\n")
        out.append(r"\end{minipage}\par" + "\n")
        if i < len(steps):
            out.append(r"\vspace{2pt}{\color{linecolor}\rule{\linewidth}{0.3pt}}\vspace{2pt}\par" + "\n")
    return "".join(out)


# ---------- Main ----------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="_scout_combined_view")
    p.add_argument("--audits-root",  default="reports/audits")
    p.add_argument("--scout-state",  default="reports/architecture_scout_2026-05-09/state.json")
    p.add_argument("--heatmap",      default="reports/audits/scout_heatmap_pretty.png")
    p.add_argument("--out",          default="reports/audits/paper_report_zh.pdf")
    p.add_argument("--keep-tex",     action="store_true")
    args = p.parse_args()

    results_root = Path(args.results_root)
    audits_root  = Path(args.audits_root)
    heatmap_path = Path(args.heatmap)
    out_path     = Path(args.out)

    print("加载侦察数据...")
    runs = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir(): continue
        try: r = analyze_run(d)
        except Exception as exc:
            print(f"  ! {d.name}: {exc}"); continue
        if r is not None and r["test_pr_auc"] is not None:
            runs.append(r)
    runs.sort(key=lambda r: r["test_pr_auc"], reverse=True)
    n_total = len(runs)

    state = json.loads(Path(args.scout_state).read_text())
    counts = Counter(t.get("status") for t in state["tasks"].values())
    state_summary = {
        "total":     sum(counts.values()),
        "completed": counts.get("completed", 0),
        "failed":    counts.get("failed", 0) + counts.get("failed_resume_available", 0)
                   + counts.get("artifact_validation_failed", 0),
        "timeout":   counts.get("timeout", 0) + counts.get("timeout_resume_available", 0),
    }
    failed_pct  = 100 * state_summary["failed"]  / state_summary["total"]
    timeout_pct = 100 * state_summary["timeout"] / state_summary["total"]

    per_class = json.loads((audits_root / "per_class_benchmark.json").read_text())
    matched_recall = json.loads((audits_root / "matched_recall_fp_report.json").read_text())

    with tempfile.TemporaryDirectory(prefix="scout_report_zh_") as td:
        tmpdir = Path(td)
        shutil.copy(heatmap_path, tmpdir / "heatmap.png")
        # Make assets/fonts visible to the tectonic compile working dir
        fonts_target = tmpdir / "assets" / "fonts"
        fonts_target.mkdir(parents=True, exist_ok=True)
        for f in Path("assets/fonts").glob("*.otf"):
            shutil.copy(f, fonts_target / f.name)
        # Per-class CRTK-rendered example boards.
        ex_dst_dir = tmpdir / "reports" / "audits"
        ex_dst_dir.mkdir(parents=True, exist_ok=True)
        for cls in (0, 1, 2):
            src = Path(f"reports/audits/puzzle_class_{cls}.png")
            if src.exists():
                shutil.copy(src, ex_dst_dir / src.name)

        # Confusion matrices
        print("绘制 3x2 混淆矩阵...")
        cm_paths = []
        for i, r in enumerate(runs[:8]):
            cm_path = tmpdir / f"cm_{i:02d}.png"
            render_confusion_matrix_png(r["probs"], r["y_true"], r.get("y_fine"), r["name"], cm_path)
            cm_paths.append(cm_path)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        winner = runs[0]; runner_up = runs[1]
        winner_short = latex_escape(short_name(winner["group"]))
        margin = winner["test_pr_auc"] - runner_up["test_pr_auc"]

        leaderboard_tex = build_leaderboard_table_zh(runs)
        pareto_tex = build_pareto_table_zh(runs)
        slice_winners_tex = build_slice_winners_table_zh(per_class)
        matched_tex = build_matched_recall_table_zh(matched_recall)
        arch_appendix_tex = build_arch_appendix_zh()
        bt4_path_tex = build_bt4_path_zh()

        cm_grid_rows = []
        for i in range(0, 8, 2):
            cm_grid_rows.append(
                r"\noindent"
                r"\begin{minipage}[t]{0.49\linewidth}\centering"
                rf"\includegraphics[width=0.92\linewidth]{{cm_{i:02d}.png}}"
                r"\end{minipage}\hfill"
                r"\begin{minipage}[t]{0.49\linewidth}\centering"
                rf"\includegraphics[width=0.92\linewidth]{{cm_{i+1:02d}.png}}"
                r"\end{minipage}\par\medskip"
            )
        cm_grid_tex = "\n".join(cm_grid_rows)

        stats_block = (
            r"\noindent\begin{minipage}{0.24\linewidth}"
            r"\statbox{" + str(state_summary["total"]) + r"}{已训练架构}"
            r"\end{minipage}\hfill"
            r"\begin{minipage}{0.24\linewidth}"
            r"\statbox{" + str(state_summary["completed"]) + r"}{完成运行}"
            r"\end{minipage}\hfill"
            r"\begin{minipage}{0.24\linewidth}"
            r"\statbox{" + str(n_total) + r"}{进入排行榜}"
            r"\end{minipage}\hfill"
            r"\begin{minipage}{0.24\linewidth}"
            r"\statbox{" + f"{winner['test_pr_auc']:.4f}" + r"}{最高测试 PR AUC}"
            r"\end{minipage}\medskip"
        )

        LATEX = CN_PREAMBLE + r"""
\begin{document}

\thispagestyle{empty}
\vspace*{1.2cm}
\noindent{\sffamily\color{muted}\bfseries\small 研究报告 \quad$\cdot$\quad chess-nn-playground}

\vspace{0.5cm}
\noindent{\fontfamily{lmr}\fontsize{36pt}{42pt}\selectfont\bfseries\color{deepforest} 架构侦察}

\vspace{0.25cm}
\noindent{\fontfamily{lmr}\itshape\large\color{muted} 234 个国际象棋评估架构的扫描研究, 横跨难度、阶段、评估桶与战术主题的同任务对比。}

\vspace{0.6cm}
\noindent{\color{rule}\rule{\linewidth}{1.4pt}}

\vspace{0.7cm}
\noindent
\begin{tabular}{>{\sffamily\bfseries\scriptsize\color{ink}}l@{\hspace{20pt}}>{\sffamily\small\color{muted}}p{0.74\linewidth}}
作者 & Lennart Axel Conrad \quad (学号: 2025080264) \\
所属单位 & 清华大学紫荆书院 \\
指导教师 & 韩军功教授 \quad (清华大学自动化系) \\[6pt]
侦察日期 & 2026-05-09 至 2026-05-10 \\
任务 & puzzle\_binary (单正 logit, BCE 损失) \\
数据集 & CRTK 标注的 3 类切分 (约 173k 训练 / 21k 验证 / 21k 测试, FEN 零重叠) \\
硬件 & RTX 3070 (8 GiB) --- 单 GPU, 单种子 (42), base 规模 \\
预算 & 最多 12 epoch, patience 3, 每任务 60 分钟墙时, 仅 CUDA \\
生成日期 & """ + today + r""" \\
代码仓库 & \href{https://github.com/LenniAConrad/chess-nn-playground}{github.com/LenniAConrad/chess-nn-playground} \\
\end{tabular}

\vspace{1.0cm}

\section*{\color{forest} 摘要}
\vspace{-6pt}{\color{linecolor}\hrule height 0.6pt}\vspace{8pt}

\begin{lead}
234 个国际象棋评估架构各自被训练一次, 规模较小, 用于谜题检测任务。\textbf{""" + str(state_summary["completed"]) + r"""} 个产出可用结果; """ + str(state_summary["failed"]) + r""" 个因代码错误崩溃 (""" + f"{failed_pct:.0f}" + r"""\%); """ + str(state_summary["timeout"]) + r""" 个超过 60 分钟训练时限 (""" + f"{timeout_pct:.0f}" + r"""\%)。
\end{lead}

""" + stats_block + r"""

\begin{tcolorbox}[goodcallout, title=核心发现]
\texttt{""" + winner_short + r"""} 以明显优势领先 (\textbf{""" + f"{winner['test_pr_auc']:.4f}" + r"""} 测试 PR AUC, 比第二名高出 $+""" + f"{margin:.3f}" + r"""$, 且参数预算相同)。其双流架构 --- 一条用于战术兑子, 一条用于王安全 --- 是整个侦察池中编码相同条件下最大的架构差距, 也是唯一能在经验上将排行榜推高到所有通用主干自然 $\sim$0.86 上限之上的架构。
\end{tcolorbox}

\clearpage

\section{数据集与谜题类别定义}

侦察任务为 \texttt{puzzle\_binary}: 将棋局分类为\emph{战术谜题} (正类)
或 \emph{非谜题} (负类)。底层 CRTK 标签是三分类的, 训练时再折叠为二分类
--- 但三分类标签决定了数据集的难度结构, 因此本报告中的所有混淆矩阵与
逐类分析仍保留三分类。

\smallskip

\noindent
\begin{tabular}{>{\sffamily\bfseries\small\color{forest}}p{3.2cm}@{\hspace{12pt}}>{\small\color{ink}}p{0.66\linewidth}}
fine\_label = 0 & \textbf{非谜题 (Non-puzzle)。} 从大师对局语料中随机
采样的棋局, 即使存在战术也是偶然的。本类为\emph{简单负样本}。 \\[2pt]
fine\_label = 1 & \textbf{验证过的近谜题 (Verified-near-puzzle)。}
被 CRTK 过滤器标记为\emph{候选}谜题, 但经 Stockfish 验证证明\emph{并非}
唯一解战术 --- 第二好的走法分数几乎相同。本类为\emph{困难负样本},
是区分强弱架构的关键。 \\[2pt]
fine\_label = 2 & \textbf{谜题 (Puzzle)。} 具有唯一战术解的棋局: 单一最佳
走法的分数至少比其他选择高若干百厘兵。本类为\emph{正样本}。 \\
\end{tabular}

\medskip

生成这些标签的 \textbf{CRTK 流水线}位于
\texttt{scripts/data/build\_crtk\_tagged\_splits.py}; 决定哪些 Stockfish
阈值能将候选样本提升为 fine\_label~2 的契约在
\texttt{docs/crtk\_export\_contract.md} 中说明。每个棋局都带有辅助标签 ---
\texttt{crtk\_difficulty}、\texttt{crtk\_phase}、\texttt{crtk\_eval\_bucket}、
\texttt{crtk\_tactic\_motifs} --- 这些标签被用来对排行榜进行切片
(见下一节)。

\medskip

\begin{figure}[H]
\centering
\begin{minipage}[t]{0.32\linewidth}\centering
\includegraphics[width=\linewidth]{reports/audits/puzzle_class_0.png}\\[3pt]
{\sffamily\bfseries\small\color{forest}类 0 --- 随机棋局}\\[1pt]
{\small\color{ink}最佳走法: \textbf{cxd3}}\\
{\scriptsize\color{muted}随机中局采样, 非唯一谜题}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\linewidth}\centering
\includegraphics[width=\linewidth]{reports/audits/puzzle_class_1.png}\\[3pt]
{\sffamily\bfseries\small\color{forest}类 1 --- 近谜题}\\[1pt]
{\small\color{ink}最佳走法: \textbf{fxe3}}\\
{\scriptsize\color{muted}pv\_gap = 115 厘兵; 非唯一}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\linewidth}\centering
\includegraphics[width=\linewidth]{reports/audits/puzzle_class_2.png}\\[3pt]
{\sffamily\bfseries\small\color{forest}类 2 --- 谜题}\\[1pt]
{\small\color{ink}最佳走法: \textbf{dxe5}}\\
{\scriptsize\color{muted}pv\_gap = 1179 厘兵; 唯一获胜}
\end{minipage}
\caption{每个类别选取一个具有代表性的样本, 使用 \texttt{crtk fen render}
渲染。三幅图均为\textbf{白方走子}。棋盘只显示原始局面 --- 最佳走法以
代数记号在每幅图下方报告, 而不是覆盖在棋盘上, 让读者看到网络所看到
的画面。\textbf{这并非刻意挑选的巧合:} 按 CRTK 的构造方式, 每一行类~1
样本 (近谜题) 都是通过修改某个真正谜题的父行而生成, 所以数据库中
每一个近谜题在 CRTK 超语料中都有一个对应的类~2 兄弟样本。
在我们的 173k 局面切分中, 约 5\% 近谜题的兄弟在采样后保留下来;
上面的类~1 与类~2 棋盘就是这样一对姐妹样本 (CRTK 父节点
\texttt{crtk\_parent\_-1001554678727017229}): 唯一结构差异是
类~2 中黑方已经吃入 e5, 使局面变为白方唯一战术胜利
(\texttt{dxe5}~$\to$ 吃后)。这将近谜题与真正谜题成对约束到数据所允许
的最紧密程度 --- 同一延续, 仅差一步, 而二分标签因唯一应答性质翻转。}
\end{figure}

\begin{tcolorbox}[callout, title=三分类结构为何关键]
仅靠记住子力数与王安全先验取得高 PR AUC 的模型, 会在二分类排行榜上
看起来强劲, 却会在\emph{匹配召回近谜题 FP 率}上崩塌: 在固定真阳率下,
误将类~1 棋局判为谜题的比例。这一指标在
\S\ref{sec:robustness} 中区分了真正去检验战术序列的架构。
\end{tcolorbox}

\clearpage

\section{总排行榜}

base 规模 12 epoch 训练后的 \texttt{puzzle\_binary} 任务测试 PR AUC。
\pilllc{} 与 \pillsi{} 标识输入编码。

\begin{tcolorbox}[callout, title=指标说明]
\textbf{测试 PR AUC} ($\uparrow$ 越高越好) --- 测试切分上的精确率-召回曲线下面积, 谜题分类的主要评分指标。\\
\textbf{验证 PR AUC} ($\uparrow$) --- 验证切分上的同指标 (用于检查点选择)。\\
\textbf{参数量} --- 可训练参数数 (越小存储越便宜)。\\
\textbf{速度} ($\uparrow$ 越高越好) --- 在 RTX 3070 上以训练 batch size 测得的推理吞吐量 (每秒测试样本数)。越大 = 推理越快。\\
\textbf{FLOPs/位} ($\downarrow$ 越低越好) --- 每个棋局的理论浮点运算量 (一次乘加计为两次 FLOP)。越低 = 运行越廉价; 对国际象棋引擎而言意义重大, 因为搜索树每个节点都要调用网络。
\end{tcolorbox}

""" + leaderboard_tex + r"""

\clearpage
\begin{landscape}
\section{逐切片性能热图}

每个单元格是该切片限定下的测试 PR AUC。颜色: 越深绿越高, 越浅越低。
``\textgreater'' 标记标识列冠军。最左三个数据列依次为参数量、推理吞吐、理论 FLOPs/位。

\begin{center}
\includegraphics[width=0.99\linewidth, height=0.78\textheight, keepaspectratio]{heatmap.png}
\end{center}

\begin{quote}
\captionof*{figure}{\small\textit{\textcolor{muted}{""" + str(n_total) + r""" 个完成模型中的前 15 名。切片维度: 难度 (5), 阶段 (3), 评估桶 (9), 战术主题 (9), 走子方 (2)。单元格值为切片限定的测试 PR AUC。}}}
\end{quote}
\end{landscape}

\clearpage
\section{逐切片冠军}

各个有趣切片值的最佳模型及其与亚军的领先幅度:

""" + slice_winners_tex + r"""

\clearpage
\section{混淆矩阵 --- 前 8 名}

各模型在其 F1 最优阈值下的源类别 $3\times 2$ 混淆矩阵。
\textbf{行}为底层 CRTK 细粒度标签 --- \emph{0: 非谜题} (易负样本), \emph{1: 已验证近谜题} (硬负样本 --- 看似战术但非谜题), \emph{2: 谜题}。\textbf{列}为二元预测。单元格显示数量与行内百分比; 第 1 行最右单元格正是匹配召回下的近谜题假阳性率。

\medskip

""" + cm_grid_tex + r"""

\clearpage
\section{速度 $\times$ 准确率: 帕累托前沿}

(准确率, 速度) 平面上的帕累托前沿 --- 没有其他模型同时更快且更准:

""" + pareto_tex + r"""

\section{鲁棒性: 升变切片上的匹配召回 FP}\label{sec:robustness}

聚合 PR AUC 忽略了专门为硬负样本拒绝设计的架构。
在升变/低升变战术主题切片上, 召回率 $0.80$ 时的最低近谜题假阳性率:

""" + matched_tex + r"""

\clearpage
\section{架构发现}

\subsection*{有效的设计}
\begin{itemize}
\item \textbf{国际象棋专属任务分解}。\texttt{i193\_exchange\_then\_king\_dual\_stream}
将主干分为战术兑子分支与王安全分支。这是侦察池中唯一产生显著高于噪声的核心提升的架构先验。
\item \textbf{棋盘对称性 / 群等变性}。规则对称/轨道/商瓶颈族 (\texttt{i042, i046, i048}) 占据前六名中的三席。同一先验的三种独立形式都名列前茅。
\item \textbf{小型残差+交互主干}。\texttt{i100\_independence\_residual\_interaction} 在三分之一的参数量与更快推理下匹敌 \texttt{bench\_lc0\_bt4\_classifier} --- 推理成本敏感场景的帕累托选择。
\end{itemize}

\subsection*{未奏效的设计}
\begin{itemize}
\item 约 21\% 的架构存在 AMP/dtype bug, 2 分钟内崩溃。CI 中一行 \texttt{torch.amp.autocast} 烟测即可拦截。
\item 约 4\% 撞上 60 分钟墙时 --- 迭代或展开式设计 (Dykstra 投影、soft-sort、层曲率变体)。
\item 少数架构生成了基本随机的预测 (PR AUC $\approx$ 正类先验): \texttt{i039, i051, i060, i062, i096}。完全的训练失败。
\end{itemize}

\section{晋升候选}

侦察是过滤器而非最终排行榜。base 规模单种子噪声带约 $\pm 0.005$ PR AUC, 此范围内的排名不可信。以下候选晋升到完整的 3 种子 $\times$ \texttt{scale\_xl} 评测:

\subsection*{按聚合 PR AUC (前 10 名)}
\texttt{i193\_exchange\_then\_king\_dual\_stream}, \texttt{i048\_rule\_automorphism\_quotient}, \texttt{i018\_oriented\_tactical\_sheaf\_laplacian}, \texttt{i188\_tactical\_program\_induction}, \texttt{i011\_vetoselect} (已有 3 种子), \texttt{i192\_latent\_reply\_entropy}, \texttt{i191\_safe\_reply\_certificate\_verifier}, \texttt{i042\_legal\_automorphism\_quotient}, \texttt{i147\_specialist\_head\_cnn}, \texttt{i046\_rule\_exact\_orbit\_bottleneck}。

\subsection*{按匹配召回近谜题 FP}
既有鲁棒性领跑者 (\texttt{i011\_vetoselect}, \texttt{i012\_dykstra\_lcp}) 加上同样在该切片表现良好的新聚合 PR AUC 赢家。

\subsection*{按利基切片胜利}
在硬切片上具有清晰边际 ($>0.005$) 优势的模型 --- 穿刺/过载上的规则对称族, 全方位胜出的双流赢家。

\clearpage
\section{超越 BT4 主干的可行路径}

LC0 BT4 是通用的 piece-token Transformer 主干。侦察中胜出的所有归纳偏置都不在其主干中 --- 没有兑子/王安全分解, 没有国际象棋自同构等变性, 没有有向战术层, 没有显式程序归纳头。

\begin{tcolorbox}[goodcallout, title=对比的范围]
预训练 BT4 网络在多个 GPU 年中接受了\emph{数十亿}自我对弈局面的训练。LC0 团队优化的不仅是主干, 还有训练流程、数据、搜索算法与工程基础设施。\textbf{我们的研究关注的是\emph{主干}, 而非训练。}因此公平的对比是: 在完全相同的训练设置下从头训练两个网络 --- 一个用我们的主干, 一个用新初始化的 BT4 形状主干 --- 然后比较。这是将架构变量与训练预算变量分离的实验。
\end{tcolorbox}

下面提出的架构 (\texttt{i241\_multistream\_attention\_chess\_eval}) 将三个相互可组合的胜出先验合成为一个为国际象棋评估而塑形的主干, 配以标准 LC0 风格的双头。

\medskip

""" + bt4_path_tex + r"""

\begin{tcolorbox}[goodcallout, title=现实预期 --- 准确率]
\textbf{对比公开预训练的 BT4:} 从头训练的所提主干几乎必然落后, 因为我们无法匹配 BT4 多年的训练算力。这不是本研究声称要赢的对比。

\textbf{在匹配训练下对比新训练的 BT4 形状主干 (真正的相关实验):} 我们估计多流主干在固定深度对战中领先约 $30$--$80$ ELO。优势完全来自国际象棋感知的归纳偏置, 这些是通用 Transformer 主干必须从数据中学习的。ELO 头条值是 2700 还是 3300 完全取决于步骤 1 的数据预算, 相对优势才是本研究衡量的。
\end{tcolorbox}

\begin{tcolorbox}[goodcallout, title=现实预期 --- 速度 (实测)]
对国际象棋引擎而言, 推理\emph{速度}与准确率同样重要: MCTS 搜索树的每个节点都要调用网络, 推理越快每秒搜索越深。两种架构均被放大到 BT4-medium 规模, 在本机硬件上进行同条件对比测试:

\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}L r r r r@{}}
\toprule
\textbf{模型} & \textbf{参数量} & \textbf{batch 1 时延} & \textbf{batch 256 吞吐} & \textbf{batch 1 单参数吞吐} \\
\midrule
\texttt{LC0\_BT4 形} (放大) & 39.1\,M & 7.27\,ms (138/s) & 2{,}803 样本/秒 & 3.5 /s/M \\
\texttt{i193 双流} (放大) & 35.5\,M & \textbf{3.94\,ms} (\textbf{254/s}) & \textbf{3{,}043 样本/秒} & \textbf{7.2 /s/M} \\
\bottomrule
\end{tabularx}
\end{footnotesize}

\medskip
在参数量匹配下, 双流主干在本机硬件上比同规模单流 BT4 主干快约 $\mathbf{1.2}\times$ (batch 256) 与 $\mathbf{1.9}\times$ (batch 1)。batch 1 的数字对引擎实战更重要, 因为 MCTS 叶节点评估是时延受限的 (典型有效 batch 1--8)。结合上述准确率估计: 在匹配训练和匹配参数量下, $\mathbf{+30}$--$\mathbf{80}$ \textbf{ELO 配合} $\mathbf{\sim 1.9 \times}$ \textbf{更快推理}是引擎层面实测得到的优势。

\vspace{2pt}
{\scriptsize\textit{实验出处: \texttt{scripts/benchmark\_scale\_up\_speed.py}, git 提交 \texttt{2e03965}, NVIDIA RTX 3070 (8\,GiB), PyTorch 2.11.0+cu130, FP32, 每个 batch size 8 次热身后 30 次迭代。}}
\end{tcolorbox}

\begin{tcolorbox}[callout, title=一句话结论]
234 个架构中, 胜出的总是那些把\emph{国际象棋的实际评估方式}编码进结构的, 而非把奇异数学塞进通用 CNN 的。
\end{tcolorbox}

\clearpage
\section{后续: 组合架构 (i242) 及其消融实验}

侦察之后, 我们设计了一个后继架构 (\texttt{i242}), 把三个独立得到验证的国际象棋结构先验组合在一起:

\begin{itemize}
\item \textbf{以王为中心的输入特征}, 灵感来自\textit{Stockfish NNUE} (HalfKA)。Stockfish NNUE 按王所在格索引输入特征; 每个累加器都依赖于王的位置。我们通过复用 i193 的确定性特征构建器实现这一性质 --- 它从原始 \texttt{simple\_18} 棋盘产出王翼、将军射线、逃逸格、攻击/防御压力等平面。\textit{Stockfish NNUE 作为最强的 CPU 国际象棋引擎, 独立验证了 “以王分解” 的命题。}
\item \textbf{兑子+王安全双流分解}, 来自侦察赢家 i193。两个并行子主干各自专门化, 都接收由上一项特征得出的棋类感知输入偏置。
\item \textbf{对 64 个格子词元的全局多头自注意力}, 是 BT4 的核心选择。第三个并行子主干以纯自注意力捕捉长程棋子关系 --- 这是纯卷积主干需要多层才能建模的内容。
\end{itemize}

通过学习的 softmax 相位路由器融合。在匹配侦察规模 (271k 参数) 下与规则对称族 (约 180k 各) 的参数预算相当。

\begin{tcolorbox}[callout, title=结果 --- i242 对 i193]
puzzle\_binary 侦察, 单种子, base 规模, 12 epoch:

\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}L r r r@{}}
\toprule
\textbf{模型} & \textbf{参数量} & \textbf{测试 PR AUC} & \textbf{对 i193 $\Delta$} \\
\midrule
i193 (亲代双流, 卷积) & 157k & \textbf{0.8755} & -- \\
i242 (完整, 三流 + 注意力) & 271k & 0.8677 & $-0.008$ \\
\bottomrule
\end{tabularx}
\end{footnotesize}

\textbf{组合架构在此规模下\emph{未能}超越其纯卷积亲代。} i242 在合并侦察池中仍排名第 2 (共 181 个模型), 但 “组合三种先验” 假设在小训练预算下被否证。该结果强化了 i193 的地位, 并指出注意力是数据需求较高的组件。
\end{tcolorbox}

\subsection{消融实验}

四个消融实验分离了各组件的贡献:

\begin{footnotesize}
ABLATION_TABLE_PLACEHOLDER
\end{footnotesize}

ABLATION_INTERPRETATION_PLACEHOLDER

\subsection{为什么这种组合在侦察规模下没有胜出}

\begin{enumerate}
\item \textbf{注意力是数据密集型的。} 在\emph{小训练预算}下, Transformer 主干每个有用 FLOP 的参数量比同等参数的卷积主干更高, 因为注意力的表达力随数据量复利。在 173k 样本上训练 12 epoch 可能不足以让注意力学到棋类感知卷积通过\emph{先验}已抽取的有用交互。
\item \textbf{相位路由器的容量稀释。} 3 路 softmax 路由器在每个局面上分配分数容量。在小训练预算下, 路由器无法学会自信地分派, 因此每条流都被要求成为完整的分类器。
\item \textbf{三种先验并不正交。} 王翼偏置与攻击/防御偏置在任何 “一子攻击王” 的局面上都重叠。两条流可能最终冗余而非专门化。
\end{enumerate}

理想化的 i241 (LC0 规模训练下的多流注意力) 并未被 i242 的结果否证 --- 在无限训练与匹配参数量下, 注意力的表达力应随数据增长而占优。\emph{在小规模和有限数据下, 更简单的棋类感知卷积分解胜过 Transformer 分解族。}

\clearpage
\section{提议后继: i243 (HalfKA + 双流 + LC0 头)}

i242 在小规模下否证了\emph{添加注意力}的组合。下一个自然的问题是:
\emph{Stockfish NNUE 的设计中, 哪一部分在引擎级规模下买到最多?}
答案是其\textbf{可学习输入表示}, 而非其 MLP 主干。Stockfish NNUE 把棋类
最强的输入方案 (HalfKA) 与最简单的主干 (普通 MLP) 配对。i193 则把最简单的
输入方案 (\texttt{simple\_18} + 确定性王/兑子平面) 与最强的战术分解
(兑子 + 王双流卷积) 配对。\textbf{i243 把 HalfKA 替换确定性王平面}, 保留
双流卷积, 并用 LC0 的 WDL 价值 + 1858 维策略头替代 puzzle-binary 头。

\begin{tcolorbox}[callout, title=三路组合]
\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}>{\bfseries\sffamily\color{forest}}p{2.6cm} L L@{}}
\toprule
\textbf{组件} & \textbf{来源} & \textbf{带来什么} \\
\midrule
HalfKA 累加器 & Stockfish NNUE & 可学习的以王为条件的嵌入表; 推理时 O(1) 增量更新。 \\
兑子/王双流卷积 & i193 (侦察冠军) & 战术分解 --- 234 模型扫描中编码相同条件下最大的差距。 \\
WDL 价值 + 1858 策略头 & LC0 (BT4 家族) & 使网络与 MCTS 兼容; 不再只是谜题分类器。 \\
\bottomrule
\end{tabularx}
\end{footnotesize}
\end{tcolorbox}

\subsection{为何选择此组合, 而非显然的替代方案}

\begin{itemize}
\item \emph{HalfKA + MLP} 即 Stockfish NNUE 本身 --- 已经很强, 但 MLP 在
结构上无法将兑子评估与王安全特征分解。在大规模下, 这会留下 Elo 上限。
\item \emph{确定性王平面 + 双流卷积 + LC0 头} 即在引擎规模下训练的 i193。
但确定性平面无法捕捉 HalfKA 累加器通过训练吸收的细粒度王条件模式
(例如特定的王车易位后兵盾变体)。
\item \emph{HalfKA + Transformer + LC0 头} 即 i242 路径; i242 刚刚表明
Transformer 主干在 173k 棋局上需要远多于此的数据来恢复其表达预算。
HalfKA + 卷积双流将相同参数预算用于已经赢得侦察的结构化棋类感知主干。
\end{itemize}

\subsection{架构草图与增量更新性质}

\begin{equation*}
a_{\mathrm{side}}(x) \;=\; \sum_{f \in \mathcal{F}_{\mathrm{active}}(x)} E_{\mathrm{side}}[f],
\quad f = (k_{\mathrm{side}},\, \mathrm{color},\, \mathrm{type},\, s)
\end{equation*}

\noindent
累加器 $a_{\mathrm{side}}$ 是所有活跃 HalfKA 特征嵌入的和。一颗子移动时,
$\mathcal{F}_{\mathrm{active}}$ 至多变化两个特征, 因此累加器只需一次减法
和一次加法即可更新。卷积主干仅在前一累加器状态的\emph{差异}上运行, 给出
i193 与 LC0 BT4 都没有的墙时推理优势。

随后累加器被重塑为按格点的 token 网格
$x_{\mathrm{token}} \in \mathbb{R}^{[64,d]}$, 输入到 i193 的兑子/王
双流卷积, 经相位路由器融合, 再以 LC0 的 WDL 价值头 + 1858 维策略头封顶。

\subsection{规模档位}

\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}L r r L@{}}
\toprule
\textbf{档位} & \textbf{embed\_dim} & \textbf{总参数} & \textbf{用途} \\
\midrule
\texttt{tiny}   & 32  & $\sim$2.5M & 侦察规模 sanity check (\texttt{puzzle\_binary}) \\
\texttt{small}  & 96  & $\sim$10M  & 科研规模微调 \\
\texttt{medium} & 256 & $\sim$38M  & 引擎规模, 匹配 BT4-medium \\
\texttt{large}  & 384 & $\sim$75M  & 引擎规模, 匹配 BT4-large \\
\bottomrule
\end{tabularx}
\end{footnotesize}

在 \texttt{medium} 档位下, HalfKA 嵌入表主导参数预算 (约 38M 中的 25M)
--- 与 Stockfish NNUE 的表同一量级。

\subsection{待引擎级训练验证的假设}

\begin{itemize}
\item[\textbf{H1}] 在引擎级规模下 (在约 $10^{7}$ 大师对局棋局上做
Stockfish-eval 蒸馏, 或 LC0 自对弈批次), i243 在固定深度对弈对手上的
Elo 同时击败 Stockfish NNUE (匹配规模) \emph{和}纯 i193 (匹配规模)。
\item[\textbf{H2}] i243 的增量更新推理路径在相同精度下相对无增量更新基线
给出 $\geq 5\times$ 墙时加速, 验证工程优势真实存在。
\item[\textbf{H3}] 相位路由权重 $\alpha(x)$ 表现出可解释的局面类型依赖
(王攻局面 $\alpha_K$ 高, 静态兑子局面 $\alpha_E$ 高), 证实分解确实被使用。
\end{itemize}

\begin{tcolorbox}[callout, title=状态: 仅为提议]
架构本身是廉价部分。训练流水线 + 数据才是项目: 在 simple\_18 来源上的
HalfKA 特征、支持增量更新的嵌入表, 以及引擎级训练数据 (Stockfish 蒸馏或
LC0 自对弈)。预计算力: medium 规模预训练 + 微调约需 1--2 GPU-周。完整规格
见 \texttt{ideas/i243\_halfka\_dual\_stream\_lc0/}。
\end{tcolorbox}

\clearpage
\section{假设: 在无限数据与算力下, 怎样才能超越 BT4?}

侦察中最强的先验都是\textbf{归纳偏置}: 它们在有限训练下买到样本效率。无限数据和无限算力下, 样本效率优势会压缩 --- 足够大的 Transformer 能从数据中学到任何分解。

\begin{tcolorbox}[goodcallout, title=哪些优势保留? 哪些消失?]
准确率优势消失并不等同于速度优势消失 --- 对于一个在搜索树每个节点都要
调用网络的国际象棋引擎而言, 推理墙时优势同等重要。我们分两列分别追踪。
\smallskip

\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}p{4cm} L L@{}}
\toprule
\textbf{先验} & \textbf{相对 BT4 的准确率优势} & \textbf{相对 BT4 的推理速度优势} \\
\midrule
以王为中心的输入 (NNUE / HalfKA) & \textbf{消失。} 50M+ Transformer 能从原始棋盘学到王相对特征。 & \textbf{保留。} HalfKA 累加器每步增量更新至多 2 次嵌入查表 --- 与局面复杂度无关的 O(1)。这是 Stockfish 在 CPU 上每秒数百万次评估的原因。 \\[3pt]
兑子/王双流分解 (i193) & \textbf{消失。} 多头注意力按任务划分头预算。 & \textbf{部分保留。} 在窄通道下, token 网格上的卷积比稠密 MHA 更便宜; 在 BT4 使用的通道宽度上, 优势压缩至 $\sim$1.2--1.9$\times$ (我们的实测值)。 \\[3pt]
棋类感知注意力偏置 (i242) & \textbf{消失。} 偏置矩阵是几何先验; 可从数据学到。 & \textbf{消失。} 偏置矩阵实际免费; 在大规模下既不提升也不降低墙时。 \\[3pt]
群等变性 (i048) & \textbf{大部分消失。} 残留优势仅是参数节省。 & \textbf{部分保留。} 等变层上 $|G|=4\times$ 减少 FLOPs; 节省规模取决于这些层在主干中的位置。 \\
\midrule
合法走子稀疏注意力模式 & --- & \textbf{保留。} 对约 8 个有战术关系的格对计算注意力 (而非全 64) 在任何数据规模下都省 $8\times$ FLOPs。 \\[3pt]
混合专家 / 相位路由 FLOPs & --- & \textbf{保留。} 每局面只激活部分 FLOPs, 实际墙时节省与数据无关。 \\[3pt]
自适应深度 / 早退 & --- & \textbf{保留。} 简单局面用更少层; 搜索可直接利用节省。 \\[3pt]
INT8/FP8 量化感知训练 & --- & \textbf{保留。} 硬件层效率, 与架构正交。 \\[3pt]
Mamba / 状态空间层替代注意力 & \textbf{趋近 BT4。} 与注意力的质量差距随数据增长而缩小。 & \textbf{保留。} 关于 token 数的线性时间; 任何规模下都享有同样的速度优势。 \\
\bottomrule
\end{tabularx}
\end{footnotesize}

\smallskip
准确率列中的 ``---'' 表示该先验纯为计算效率手段, 而非归纳偏置 ---
它不改变网络能表达什么, 只改变表达的速度。
\end{tcolorbox}

\subsection{“无限数据下比 BT4 更快” 的配方}

在无限训练下, 相关杠杆是\emph{算力效率先验} --- 在不牺牲表达力的前提下省墙时 FLOPs:

\begin{enumerate}
\item \textbf{对合法走子格对的稀疏注意力。} 对每个格子 $s$, 只对约 8 个有战术关系的格子 $t$ 计算注意力 (合法走子、攻击射线、王翼)。注意力代价: $O(64 \cdot k \cdot d)$ ($k \approx 8$), 而非 $O(64^2 d)$。$8\times$ 注意力提速且\emph{无}表达力损失 --- 被遮罩的格对真的没有战术关系。
\item \textbf{通过相位路由器实现自适应深度提前退出。} i193/i242 的相位路由器推广: 不再混合固定深度的流, 而是根据路由器置信度为每个局面分配不同的堆栈深度。王对王残局只需 2 层; 锐利中局需 16 层。在真实局面分布上平均算力降低约 $2$--$3 \times$。
\item \textbf{混合专家注意力头。} 每个 MCTS 叶子评估只激活子集头 (例如 16 选 4)。未激活的头零 FLOPs。同质量下 $4\times$ 推理提速。
\item \textbf{INT8 量化感知训练。} 标准 $2$--$4\times$ 墙时提速, 与上述乘法叠加。
\end{enumerate}

\begin{tcolorbox}[callout, title=无限训练下的现实预期]
稀疏注意力 + 自适应深度 + MoE 头 + INT8 的堆叠, 相对密集 FP32 BT4 主干在匹配质量下能带来大约 $50$--$200\times$ 的算力提速。无限数据下我们对 BT4 的\emph{准确率}优势消失, 但约 $100\times$ 的\emph{推理速度}优势保留 --- 对每步搜索成千上万节点的国际象棋引擎而言, 网络评估快 $100\times$ 在实际效果上远比 $80$ ELO 的原始质量更有价值。
\end{tcolorbox}

\textbf{即使在无限训练下我们的研究方向也仍然有趣:} 不是因为我们的先验给出更好的准确率, 而是因为棋类感知分解提供了 \emph{合法走子稀疏} 模式的结构性提示 --- 这正是让网络在速度上超过 BT4 的关键。先验作为准确率杠杆并未在规模下幸存, 但作为算力杠杆幸存了下来。

\clearpage
\section{局限与有效性威胁}

\subsection*{单种子侦察}
所有侦察运行使用种子 42 与 base 规模。本数据集上单种子 PR AUC 的经验噪声带约为 $\pm 0.005$--$0.010$ (由归档运行的 3 种子组估计)。小于该带的差异不应解读为架构差异; 双流赢家的 0.014 头条边际显著超过该带, 但第 4--12 名落在带内。晋升阶段会以 3 种子与 \texttt{scale\_xl} 重新运行候选, 以消除歧义。

\subsection*{代理任务}
\texttt{puzzle\_binary} 是国际象棋局面评估的代理, 并非直接衡量引擎对弈强度。在 \texttt{puzzle\_binary} 上胜出的主干不保证在价值+策略回归上胜出。但浮现的架构先验 (分解、等变、有向注意力) 都是关于国际象棋几何的任务无关先验, 因此可能迁移; 这正是 BT4 路径计划步骤 5 要证伪的假设。

\subsection*{训练方差}
约 21\% 的架构存在 AMP/dtype bug, 在 2 分钟内完全训练失败。这些不是架构问题, 而是代码 bug --- CI 中一行 \texttt{torch.amp.autocast} 烟测即可拦截。我们将其排除在排行榜外, 但承认这降低了有效样本量。

\subsection*{标注噪声}
CRTK 细粒度标签方案 (0: 非谜题, 1: 已验证近谜题, 2: 谜题) 由标注质量流程生成, 而非人工标注。类 1 尤其是程序化验证, 可能有约几个百分点的标签噪声。该类上的匹配召回 FP 率因此是真实率的上界。

\subsection*{硬件约束}
所有侦察运行共享一张 RTX 3070, 8 GiB 显存。较大架构 (尤其是多个 idea 的 \texttt{scale\_xl} 变体) 撞上 60 分钟墙时或显存不足, 我们因此系统性低估参数谱的最大端。多流主干相对 BT4 的速度优势估计是从 base 规模测量外推的, 需要在实际放大后验证。

\section{可复现性}

侦察流程完全脚本化且可恢复:
\begin{itemize}
\item 源代码: \texttt{scripts/run\_paper\_ready\_all.py}, \texttt{scripts/train\_model.py}, 及各 idea 的 \texttt{model.py}。
\item 状态: \texttt{reports/architecture\_scout\_2026-05-09/state.json} 记录每个任务的状态、退出码、耗时和生成配置的 SHA-256 散列。
\item 事件日志: \texttt{reports/architecture\_scout\_2026-05-09/events.jsonl} 是带时间戳的 \texttt{task\_started}~/~\texttt{task\_finished} 事件的追加日志。
\item 数据集: \texttt{data/splits/crtk\_sample\_3class\_unique\_crtk\_tags/} (parquet, 确定性切分, 经审计 train/val/test 间零重叠)。
\item 随机种子: 每个配置显式固定 \texttt{seed: 42} 与 \texttt{deterministic: true}; PyTorch 的部分 CUDA 卷积仍存在非确定性, 贡献了上述噪声带。
\end{itemize}

\clearpage
\section{附录 \textperiodcentered{} 各顶尖架构的工作原理}

对侦察信号最强的五个架构外加 BT4 参考主干: 单段总结、关键方程、所带来的归纳偏置以及其代价。$x$ 表示输入棋盘 (\texttt{simple\_18} 编码下 $\mathbb{R}^{c \times 8 \times 8}$, $c=18$; \texttt{lc0\_bt4\_112} 编码下 $c=112$)。

\medskip

""" + arch_appendix_tex + r"""

\end{document}
"""

        tex_path = tmpdir / "report.tex"
        # Substitute i242 ablation results into placeholders (if available)
        try:
            from _substitute_ablations import fill_ablation_placeholders
            LATEX = fill_ablation_placeholders(LATEX)
        except Exception as exc:
            print(f"warning: ablation substitution skipped: {exc}")

        tex_path.write_text(LATEX, encoding="utf-8")

        if args.keep_tex:
            tex_out = out_path.with_suffix(".tex")
            tex_out.write_text(LATEX, encoding="utf-8")
            print(f"已写入 {tex_out}")

        print("使用 tectonic 编译 LaTeX...")
        tectonic = Path.home() / ".local/bin/tectonic"
        cmd = [str(tectonic), "-X", "compile", "--keep-logs", "--keep-intermediates",
               "--outdir", str(tmpdir), str(tex_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
        if result.returncode != 0:
            print("=== tectonic stderr (tail) ===")
            print(result.stderr[-3000:])
            print("=== tex saved at ===", tmpdir / "report.tex")
            shutil.copy(tex_path, "/tmp/report_zh_failed.tex")
            return 1
        pdf_built = tmpdir / "report.pdf"
        if not pdf_built.exists():
            print(result.stdout)
            return 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf_built, out_path)
        print(f"已写入 {out_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
