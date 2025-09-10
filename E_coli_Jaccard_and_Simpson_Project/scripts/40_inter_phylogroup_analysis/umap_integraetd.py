import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from PIL import Image
import matplotlib.patches as patches
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None  # Disable pixel limit for high-resolution images

# ============================================================
# CONFIGURATION (constants)
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

FIGURE_OUTPUT_DIR = ROOT / "Figures"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FIGURE_DIR = ROOT/ "figures"/ "umap"

DPI_PDF = 300   # Resolution for reading PDFs

# ============================================================
# LOAD PANEL FIGURES
# ============================================================
umap_amr_jaccard = convert_from_path(str(INPUT_FIGURE_DIR / "umap_projection_amr_jaccard.pdf"), dpi=DPI_PDF)[0]
umap_amr_simpson = convert_from_path(str(INPUT_FIGURE_DIR / "umap_projection_amr_simpson.pdf"), dpi=DPI_PDF)[0]
umap_vf_jaccard  = convert_from_path(str(INPUT_FIGURE_DIR / "umap_projection_vf_jaccard.pdf"), dpi=DPI_PDF)[0]
umap_vf_simpson  = convert_from_path(str(INPUT_FIGURE_DIR / "umap_projection_vf_simpson_1st.pdf"), dpi=DPI_PDF)[0]

# Load pie charts
pie1_amr_simpson = convert_from_path(str(INPUT_FIGURE_DIR / "cluster1_amr_simpson_pie_chart.pdf"), dpi=DPI_PDF)[0]
pie2_amr_simpson = convert_from_path(str(INPUT_FIGURE_DIR / "cluster2_amr_simpson_pie_chart.pdf"), dpi=DPI_PDF)[0]
pie3_amr_simpson = convert_from_path(str(INPUT_FIGURE_DIR / "cluster3_amr_simpson_pie_chart.pdf"), dpi=DPI_PDF)[0]
pie4_amr_simpson = convert_from_path(str(INPUT_FIGURE_DIR / "cluster4_amr_simpson_pie_chart.pdf"), dpi=DPI_PDF)[0]

pie1_vf_simpson = convert_from_path(str(INPUT_FIGURE_DIR / "non_b2_cluster_vf_simpson_pie_chart.pdf"), dpi=DPI_PDF)[0]
pie2_vf_simpson = convert_from_path(str(INPUT_FIGURE_DIR / "b2_cluster_vf_simpson_pie_chart.pdf"), dpi=DPI_PDF)[0]

# Load legend image
legend_image = convert_from_path(str(INPUT_FIGURE_DIR / "umap_legend.pdf"), dpi=DPI_PDF)[0]

# ============================================================
# CREATE INTEGRATED FIGURE
# ============================================================
fig = plt.figure(figsize=(30, 24))

# Main UMAP panel positions
ax_a = fig.add_axes([0.01, 0.51, 0.445, 0.43])
ax_b = fig.add_axes([0.47, 0.51, 0.445, 0.43])
ax_c = fig.add_axes([0.01, 0.02, 0.445, 0.43])
ax_d = fig.add_axes([0.47, 0.02, 0.445, 0.43])

# Display UMAP panels
for ax, img in zip(
    [ax_a, ax_b, ax_c, ax_d],
    [umap_amr_jaccard, umap_amr_simpson, umap_vf_jaccard, umap_vf_simpson]
):
    ax.imshow(img)
    ax.axis('off')
    rect = patches.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        linewidth=4, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

# Panel labels (a–d)
label_positions = [(0.01, 0.95), (0.47, 0.95), (0.01, 0.46), (0.47, 0.46)]
labels = ['a', 'b', 'c', 'd']
for (x, y), label in zip(label_positions, labels):
    fig.text(x, y, label, fontsize=75, fontname='Arial', weight='bold')

# Add pie charts
for pos, img in [
    ([0.47, 0.88, 0.05, 0.05], pie1_amr_simpson),
    ([0.65, 0.88, 0.05, 0.05], pie2_amr_simpson),
    ([0.865, 0.88, 0.05, 0.05], pie3_amr_simpson),
    ([0.865, 0.53, 0.05, 0.05], pie4_amr_simpson),
    ([0.565, 0.175, 0.05, 0.05], pie1_vf_simpson),
    ([0.780, 0.288, 0.05, 0.05], pie2_vf_simpson)
]:
    ax_pie = fig.add_axes(pos)
    ax_pie.imshow(img)
    ax_pie.axis('off')

# Add legend
legend_ax = fig.add_axes([0.93, 0.49, 0.08, 0.34])
legend_ax.imshow(legend_image)
legend_ax.axis('off')

fig.text(0.92, 0.8, 'Phylogroup', fontsize=50, fontname='Arial', weight='bold')

# Column and row titles
fig.text(0.10, 0.97, 'AMR Jaccard', ha='center', va='top', fontsize=45, fontname='Arial')
fig.text(0.57, 0.97, 'AMR Simpson', ha='center', va='top', fontsize=45, fontname='Arial')
fig.text(0.10, 0.48, 'VF Jaccard', ha='center', va='top', fontsize=45, fontname='Arial')
fig.text(0.57, 0.48, 'VF Simpson', ha='center', va='top', fontsize=45, fontname='Arial')

# Save output
plt.savefig(OUT_FIGURE = FIGURE_OUTPUT_DIR / "umap_projection_integrated.pdf", dpi=600, bbox_inches='tight')
