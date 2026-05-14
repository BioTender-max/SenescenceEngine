"""
SenescenceEngine: Cellular Senescence Analysis Pipeline
- Senescence scoring (SenMayo gene set)
- Telomere length estimation (TL-score)
- SASP factor analysis
- Cell cycle arrest detection (p21/p16 axis)
- Senescence trajectory analysis
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 60)
print("SenescenceEngine v1.0")
print("Cellular Senescence Analysis Pipeline")
print("=" * 60)

# ─── 1. SYNTHETIC SINGLE-CELL DATA ───────────────────────────
N_CELLS = 800
N_GENES = 2000

# Cell types: proliferating, quiescent, senescent, SASP-high
cell_types = np.array(
    ['proliferating'] * 250 +
    ['quiescent'] * 200 +
    ['senescent'] * 200 +
    ['sasp_high'] * 150
)
n_cells = len(cell_types)

# Gene expression matrix
expr = np.random.lognormal(mean=1.5, sigma=1.2, size=(n_cells, N_GENES))

# ─── 2. SENESCENCE GENE SETS ─────────────────────────────────
# SenMayo-inspired gene set (Saul et al. 2022)
SENESCENCE_GENES = [
    'CDKN1A',  # p21
    'CDKN2A',  # p16
    'TP53',
    'RB1',
    'LMNB1',   # down in senescence
    'MKI67',   # down in senescence
    'PCNA',    # down in senescence
    'GLB1',    # SA-beta-gal
    'H2AFX',   # gamma-H2AX (DNA damage)
    'HMGB1',
    'HMGB2',
]

SASP_GENES = [
    'IL6',
    'IL8',     # CXCL8
    'IL1A',
    'IL1B',
    'TNF',
    'CXCL1',
    'CXCL2',
    'MMP1',
    'MMP3',
    'MMP9',
    'MMP13',
    'VEGFA',
    'CCL2',
    'CCL5',
    'IGFBP3',
    'IGFBP7',
    'PAI1',    # SERPINE1
    'GDF15',
]

CELL_CYCLE_GENES = [
    'CDK4',
    'CDK6',
    'CCND1',
    'CCNE1',
    'CCNA2',
    'CCNB1',
    'E2F1',
    'E2F3',
    'MCM2',
    'MCM6',
]

TELOMERE_GENES = [
    'TERT',
    'TERC',
    'TINF2',
    'POT1',
    'TPP1',
    'TRF1',
    'TRF2',
    'RAP1',
    'RTEL1',
]

all_gene_sets = {
    'Senescence': SENESCENCE_GENES,
    'SASP': SASP_GENES,
    'Cell_Cycle': CELL_CYCLE_GENES,
    'Telomere': TELOMERE_GENES,
}

# Assign gene indices
gene_names = [f'GENE{i:04d}' for i in range(N_GENES)]
gene_set_indices = {}
for gs_name, gs_genes in all_gene_sets.items():
    idxs = list(range(len(gene_set_indices) * 50,
                      len(gene_set_indices) * 50 + len(gs_genes)))
    gene_set_indices[gs_name] = idxs
    for j, g in enumerate(gs_genes):
        gene_names[idxs[j]] = g

# Modulate expression by cell type
type_to_idx = {t: np.where(cell_types == t)[0] for t in np.unique(cell_types)}

# Senescent cells: high p21/p16, low Ki67/PCNA, high gamma-H2AX
sen_idx = type_to_idx['senescent']
sasp_idx = type_to_idx['sasp_high']
prol_idx = type_to_idx['proliferating']
quies_idx = type_to_idx['quiescent']

# Senescence markers up
for gi in gene_set_indices['Senescence'][:8]:
    expr[sen_idx, gi] *= np.random.uniform(3, 8, size=len(sen_idx))
    expr[sasp_idx, gi] *= np.random.uniform(2, 5, size=len(sasp_idx))
# Proliferation markers down in senescent
for gi in gene_set_indices['Senescence'][4:7]:  # LMNB1, MKI67, PCNA
    expr[sen_idx, gi] *= np.random.uniform(0.1, 0.3, size=len(sen_idx))

# SASP genes up in sasp_high and senescent
for gi in gene_set_indices['SASP']:
    expr[sasp_idx, gi] *= np.random.uniform(5, 15, size=len(sasp_idx))
    expr[sen_idx, gi] *= np.random.uniform(2, 6, size=len(sen_idx))

# Cell cycle genes down in senescent/quiescent
for gi in gene_set_indices['Cell_Cycle']:
    expr[sen_idx, gi] *= np.random.uniform(0.05, 0.2, size=len(sen_idx))
    expr[quies_idx, gi] *= np.random.uniform(0.2, 0.5, size=len(quies_idx))

# Telomere genes down in senescent
for gi in gene_set_indices['Telomere']:
    expr[sen_idx, gi] *= np.random.uniform(0.1, 0.4, size=len(sen_idx))

print(f"\n[Data] {n_cells} cells, {N_GENES} genes")
print(f"  Cell types: {dict(zip(*np.unique(cell_types, return_counts=True)))}")

# ─── 3. SENESCENCE SCORING (ssGSEA-style) ────────────────────
def ssgsea_score(expr_matrix, gene_indices):
    """Single-sample GSEA score for a gene set."""
    scores = []
    for cell_expr in expr_matrix:
        ranked = np.argsort(cell_expr)[::-1]
        rank_pos = np.where(np.isin(ranked, gene_indices))[0]
        n_genes = len(gene_indices)
        n_total = len(cell_expr)
        # Running sum
        hit_indicator = np.zeros(n_total)
        hit_indicator[rank_pos] = 1
        running_sum = np.cumsum(hit_indicator / n_genes - (1 - hit_indicator) / (n_total - n_genes))
        score = np.max(running_sum) - np.min(running_sum)
        scores.append(score)
    return np.array(scores)

print("\n[Scoring] Computing senescence scores (ssGSEA)...")
sen_scores = ssgsea_score(expr, gene_set_indices['Senescence'])
sasp_scores = ssgsea_score(expr, gene_set_indices['SASP'])
cc_scores = ssgsea_score(expr, gene_set_indices['Cell_Cycle'])
tel_scores = ssgsea_score(expr, gene_set_indices['Telomere'])

# Composite senescence score
composite_score = (sen_scores + sasp_scores - cc_scores - tel_scores) / 4

print(f"  Senescence score by cell type:")
for ct in ['proliferating', 'quiescent', 'senescent', 'sasp_high']:
    idx = type_to_idx[ct]
    print(f"    {ct:15s}: mean={composite_score[idx].mean():.3f} ± {composite_score[idx].std():.3f}")

# ─── 4. TELOMERE LENGTH ESTIMATION ───────────────────────────
print("\n[Telomere] Estimating telomere length scores...")

# TL-score: ratio of telomere gene expression to reference genes
tel_gene_expr = expr[:, gene_set_indices['Telomere']].mean(axis=1)
ref_gene_expr = expr[:, 500:550].mean(axis=1)  # housekeeping genes
tl_score = np.log2(tel_gene_expr / (ref_gene_expr + 1e-6))

# Normalize to 0-100
tl_score_norm = (tl_score - tl_score.min()) / (tl_score.max() - tl_score.min()) * 100

print(f"  Telomere length score by cell type:")
for ct in ['proliferating', 'quiescent', 'senescent', 'sasp_high']:
    idx = type_to_idx[ct]
    print(f"    {ct:15s}: TL-score={tl_score_norm[idx].mean():.1f} ± {tl_score_norm[idx].std():.1f}")

# ─── 5. SASP FACTOR ANALYSIS ─────────────────────────────────
print("\n[SASP] Analyzing SASP factor expression...")

sasp_expr = expr[:, gene_set_indices['SASP']]
sasp_gene_names = SASP_GENES

# Mean SASP expression per cell type
sasp_by_type = {}
for ct in ['proliferating', 'quiescent', 'senescent', 'sasp_high']:
    idx = type_to_idx[ct]
    sasp_by_type[ct] = sasp_expr[idx].mean(axis=0)

# Top SASP factors in senescent vs proliferating
sasp_fc = np.log2((sasp_by_type['sasp_high'] + 1) / (sasp_by_type['proliferating'] + 1))
top_sasp_idx = np.argsort(sasp_fc)[::-1][:5]
print(f"  Top SASP factors (sasp_high vs proliferating):")
for i in top_sasp_idx:
    print(f"    {sasp_gene_names[i]:10s}: log2FC={sasp_fc[i]:.2f}")

# ─── 6. CELL CYCLE ARREST DETECTION ─────────────────────────
print("\n[Cell Cycle] Detecting cell cycle arrest...")

# p21/p16 axis
p21_idx = gene_set_indices['Senescence'][0]  # CDKN1A
p16_idx = gene_set_indices['Senescence'][1]  # CDKN2A
ki67_idx = gene_set_indices['Senescence'][5]  # MKI67

p21_expr = expr[:, p21_idx]
p16_expr = expr[:, p16_idx]
ki67_expr = expr[:, ki67_idx]

# Classify cells by arrest status
p21_thresh = np.percentile(p21_expr, 75)
p16_thresh = np.percentile(p16_expr, 75)
ki67_thresh = np.percentile(ki67_expr, 25)

arrested_p21 = (p21_expr > p21_thresh) & (ki67_expr < ki67_thresh)
arrested_p16 = (p16_expr > p16_thresh) & (ki67_expr < ki67_thresh)
arrested_both = arrested_p21 & arrested_p16

print(f"  p21-arrested cells: {arrested_p21.sum()} ({100*arrested_p21.mean():.1f}%)")
print(f"  p16-arrested cells: {arrested_p16.sum()} ({100*arrested_p16.mean():.1f}%)")
print(f"  Dual-arrested cells: {arrested_both.sum()} ({100*arrested_both.mean():.1f}%)")

# Overlap with senescent cells
sen_mask = cell_types == 'senescent'
print(f"  Senescent cells with p21 arrest: {(arrested_p21 & sen_mask).sum()}/{sen_mask.sum()}")

# ─── 7. SENESCENCE TRAJECTORY (PSEUDOTIME) ───────────────────
print("\n[Trajectory] Computing senescence trajectory...")

# Simple PCA-based pseudotime
from numpy.linalg import svd

# Use top variable genes
gene_var = expr.var(axis=0)
top_var_idx = np.argsort(gene_var)[::-1][:200]
expr_sub = expr[:, top_var_idx]
expr_centered = expr_sub - expr_sub.mean(axis=0)

# SVD
U, S, Vt = svd(expr_centered, full_matrices=False)
pca_coords = U[:, :2] * S[:2]

# Pseudotime: project onto PC1 (proliferating → senescent axis)
# Root: most proliferating cell
root_idx = prol_idx[np.argmin(pca_coords[prol_idx, 0])]
pseudotime = pca_coords[:, 0] - pca_coords[root_idx, 0]
pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())

print(f"  Pseudotime by cell type:")
for ct in ['proliferating', 'quiescent', 'senescent', 'sasp_high']:
    idx = type_to_idx[ct]
    print(f"    {ct:15s}: pseudotime={pseudotime[idx].mean():.3f} ± {pseudotime[idx].std():.3f}")

# ─── 8. DIFFERENTIAL EXPRESSION: SENESCENT vs PROLIFERATING ──
print("\n[DE] Differential expression: senescent vs proliferating...")

pval_list = []
fc_list = []
for g in range(N_GENES):
    g_sen = expr[sen_idx, g]
    g_prol = expr[prol_idx, g]
    t, p = stats.ttest_ind(np.log1p(g_sen), np.log1p(g_prol))
    fc = np.log2(g_sen.mean() / (g_prol.mean() + 1e-6))
    pval_list.append(p)
    fc_list.append(fc)

pvals = np.array(pval_list)
fcs = np.array(fc_list)

# BH FDR
from scipy.stats import rankdata
n = len(pvals)
ranks = rankdata(pvals)
fdr = np.minimum(1, pvals * n / ranks)
# Ensure monotonicity
for i in np.argsort(pvals)[::-1]:
    if i < n - 1:
        fdr[np.argsort(pvals)[np.where(np.argsort(pvals) == i)[0][0] + 1]] = min(
            fdr[np.argsort(pvals)[np.where(np.argsort(pvals) == i)[0][0] + 1]],
            fdr[i]
        )

sig_mask = (fdr < 0.05) & (np.abs(fcs) > 1)
n_up = ((fcs > 1) & (fdr < 0.05)).sum()
n_down = ((fcs < -1) & (fdr < 0.05)).sum()
print(f"  Significant DE genes: {sig_mask.sum()} (up={n_up}, down={n_down})")

# Top upregulated
top_up = np.argsort(fcs * (fdr < 0.05))[::-1][:5]
print(f"  Top upregulated in senescent:")
for i in top_up:
    print(f"    {gene_names[i]:12s}: log2FC={fcs[i]:.2f}, FDR={fdr[i]:.2e}")

# ─── 9. VISUALIZATION ────────────────────────────────────────
print("\n[Viz] Generating dashboard...")

colors = {'proliferating': '#2196F3', 'quiescent': '#4CAF50',
          'senescent': '#FF5722', 'sasp_high': '#9C27B0'}
ct_colors = np.array([colors[ct] for ct in cell_types])

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0a0a0a')
gs_main = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

# Panel 1: PCA colored by cell type
ax1 = fig.add_subplot(gs_main[0, 0])
ax1.set_facecolor('#111111')
for ct, col in colors.items():
    idx = type_to_idx[ct]
    ax1.scatter(pca_coords[idx, 0], pca_coords[idx, 1],
                c=col, s=8, alpha=0.6, label=ct)
ax1.set_xlabel('PC1', color='white', fontsize=9)
ax1.set_ylabel('PC2', color='white', fontsize=9)
ax1.set_title('PCA: Cell States', color='white', fontsize=10, fontweight='bold')
ax1.tick_params(colors='white', labelsize=7)
for spine in ax1.spines.values():
    spine.set_color('#333333')
ax1.legend(fontsize=6, facecolor='#222222', labelcolor='white', markerscale=2)

# Panel 2: Composite senescence score
ax2 = fig.add_subplot(gs_main[0, 1])
ax2.set_facecolor('#111111')
ct_order = ['proliferating', 'quiescent', 'senescent', 'sasp_high']
bp_data = [composite_score[type_to_idx[ct]] for ct in ct_order]
bp = ax2.boxplot(bp_data, patch_artist=True, notch=False)
for patch, ct in zip(bp['boxes'], ct_order):
    patch.set_facecolor(colors[ct])
    patch.set_alpha(0.8)
for element in ['whiskers', 'caps', 'medians', 'fliers']:
    for item in bp[element]:
        item.set_color('white')
ax2.set_xticklabels([ct[:5] for ct in ct_order], color='white', fontsize=8)
ax2.set_ylabel('Composite Score', color='white', fontsize=9)
ax2.set_title('Senescence Score by Cell Type', color='white', fontsize=10, fontweight='bold')
ax2.tick_params(colors='white', labelsize=7)
for spine in ax2.spines.values():
    spine.set_color('#333333')

# Panel 3: Telomere length score
ax3 = fig.add_subplot(gs_main[0, 2])
ax3.set_facecolor('#111111')
tl_data = [tl_score_norm[type_to_idx[ct]] for ct in ct_order]
bp3 = ax3.boxplot(tl_data, patch_artist=True)
for patch, ct in zip(bp3['boxes'], ct_order):
    patch.set_facecolor(colors[ct])
    patch.set_alpha(0.8)
for element in ['whiskers', 'caps', 'medians', 'fliers']:
    for item in bp3[element]:
        item.set_color('white')
ax3.set_xticklabels([ct[:5] for ct in ct_order], color='white', fontsize=8)
ax3.set_ylabel('TL-Score (0-100)', color='white', fontsize=9)
ax3.set_title('Telomere Length Score', color='white', fontsize=10, fontweight='bold')
ax3.tick_params(colors='white', labelsize=7)
for spine in ax3.spines.values():
    spine.set_color('#333333')

# Panel 4: SASP heatmap
ax4 = fig.add_subplot(gs_main[1, 0])
ax4.set_facecolor('#111111')
sasp_matrix = np.array([sasp_by_type[ct] for ct in ct_order])
sasp_norm = (sasp_matrix - sasp_matrix.min(axis=0)) / (sasp_matrix.max(axis=0) - sasp_matrix.min(axis=0) + 1e-6)
im = ax4.imshow(sasp_norm, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
ax4.set_xticks(range(len(SASP_GENES)))
ax4.set_xticklabels(SASP_GENES, rotation=90, fontsize=5, color='white')
ax4.set_yticks(range(4))
ax4.set_yticklabels([ct[:8] for ct in ct_order], color='white', fontsize=8)
ax4.set_title('SASP Factor Expression', color='white', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

# Panel 5: Pseudotime trajectory
ax5 = fig.add_subplot(gs_main[1, 1])
ax5.set_facecolor('#111111')
sc = ax5.scatter(pca_coords[:, 0], pca_coords[:, 1],
                 c=pseudotime, cmap='plasma', s=8, alpha=0.7)
plt.colorbar(sc, ax=ax5, fraction=0.046, pad=0.04, label='Pseudotime').ax.yaxis.set_tick_params(color='white', labelcolor='white')
ax5.set_xlabel('PC1', color='white', fontsize=9)
ax5.set_ylabel('PC2', color='white', fontsize=9)
ax5.set_title('Senescence Pseudotime', color='white', fontsize=10, fontweight='bold')
ax5.tick_params(colors='white', labelsize=7)
for spine in ax5.spines.values():
    spine.set_color('#333333')

# Panel 6: Volcano plot
ax6 = fig.add_subplot(gs_main[1, 2])
ax6.set_facecolor('#111111')
neg_log_p = -np.log10(pvals + 1e-300)
colors_v = np.where((fcs > 1) & (fdr < 0.05), '#FF5722',
           np.where((fcs < -1) & (fdr < 0.05), '#2196F3', '#555555'))
ax6.scatter(fcs, neg_log_p, c=colors_v, s=3, alpha=0.6)
ax6.axvline(x=1, color='#FF5722', linestyle='--', alpha=0.5, linewidth=0.8)
ax6.axvline(x=-1, color='#2196F3', linestyle='--', alpha=0.5, linewidth=0.8)
ax6.axhline(y=-np.log10(0.05), color='yellow', linestyle='--', alpha=0.5, linewidth=0.8)
ax6.set_xlabel('log2FC (Senescent/Proliferating)', color='white', fontsize=9)
ax6.set_ylabel('-log10(p-value)', color='white', fontsize=9)
ax6.set_title(f'Volcano: {sig_mask.sum()} DE Genes', color='white', fontsize=10, fontweight='bold')
ax6.tick_params(colors='white', labelsize=7)
for spine in ax6.spines.values():
    spine.set_color('#333333')

# Panel 7: Cell cycle arrest
ax7 = fig.add_subplot(gs_main[2, 0])
ax7.set_facecolor('#111111')
arrest_counts = {
    'p21-only': (arrested_p21 & ~arrested_p16).sum(),
    'p16-only': (arrested_p16 & ~arrested_p21).sum(),
    'Dual': arrested_both.sum(),
    'None': (~arrested_p21 & ~arrested_p16).sum(),
}
bars = ax7.bar(arrest_counts.keys(), arrest_counts.values(),
               color=['#FF9800', '#E91E63', '#9C27B0', '#607D8B'], alpha=0.85)
ax7.set_ylabel('Cell Count', color='white', fontsize=9)
ax7.set_title('Cell Cycle Arrest Classification', color='white', fontsize=10, fontweight='bold')
ax7.tick_params(colors='white', labelsize=8)
for spine in ax7.spines.values():
    spine.set_color('#333333')
for bar in bars:
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(int(bar.get_height())), ha='center', va='bottom', color='white', fontsize=8)

# Panel 8: Score correlations
ax8 = fig.add_subplot(gs_main[2, 1])
ax8.set_facecolor('#111111')
ax8.scatter(sen_scores, sasp_scores, c=ct_colors, s=6, alpha=0.5)
ax8.set_xlabel('Senescence Score', color='white', fontsize=9)
ax8.set_ylabel('SASP Score', color='white', fontsize=9)
r, p = stats.pearsonr(sen_scores, sasp_scores)
ax8.set_title(f'Senescence vs SASP (r={r:.2f})', color='white', fontsize=10, fontweight='bold')
ax8.tick_params(colors='white', labelsize=7)
for spine in ax8.spines.values():
    spine.set_color('#333333')

# Panel 9: Summary stats
ax9 = fig.add_subplot(gs_main[2, 2])
ax9.set_facecolor('#111111')
ax9.axis('off')
summary = [
    f"SenescenceEngine v1.0",
    f"",
    f"Cells analyzed: {n_cells}",
    f"Genes profiled: {N_GENES}",
    f"",
    f"Senescent cells: {(cell_types=='senescent').sum()}",
    f"SASP-high cells: {(cell_types=='sasp_high').sum()}",
    f"",
    f"DE genes (FDR<0.05): {sig_mask.sum()}",
    f"  Upregulated: {n_up}",
    f"  Downregulated: {n_down}",
    f"",
    f"p21-arrested: {arrested_p21.sum()} ({100*arrested_p21.mean():.0f}%)",
    f"p16-arrested: {arrested_p16.sum()} ({100*arrested_p16.mean():.0f}%)",
    f"",
    f"Top SASP: {SASP_GENES[top_sasp_idx[0]]}",
    f"  log2FC={sasp_fc[top_sasp_idx[0]]:.2f}",
]
for i, line in enumerate(summary):
    color = '#E9ED4C' if i == 0 else ('white' if line else '#555555')
    ax9.text(0.05, 0.97 - i * 0.055, line, transform=ax9.transAxes,
             color=color, fontsize=8.5, va='top',
             fontweight='bold' if i == 0 else 'normal')

fig.suptitle('SenescenceEngine: Cellular Senescence Analysis Dashboard',
             color='white', fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/workspace/senescence_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0a')
plt.close()
print("  Dashboard saved.")

print("\n" + "=" * 60)
print("SenescenceEngine COMPLETE")
print(f"  Cells: {n_cells} | Genes: {N_GENES}")
print(f"  Senescent cells scored: {(cell_types=='senescent').sum()}")
print(f"  DE genes: {sig_mask.sum()} (up={n_up}, down={n_down})")
print(f"  Top SASP factor: {SASP_GENES[top_sasp_idx[0]]} (log2FC={sasp_fc[top_sasp_idx[0]]:.2f})")
print(f"  p21-arrested: {arrested_p21.sum()} cells")
print(f"  Senescence-SASP correlation: r={r:.3f}")
print("=" * 60)
