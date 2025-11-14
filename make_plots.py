import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from photoz_utils import get_point_metrics
from matplotlib.lines import Line2D

def get_predictions(file_name):
    """
    Load predictions from CSV files for a given file name.
    Args:
        file_name (str): The name of the file to load predictions from.
    Returns:
        tuple: DataFrames containing predictions from GalaxiesML, TransferZ, and Combo models.
    """
    if not file_name:
        raise ValueError("file_name must be provided to load predictions.")
    # Ensure the directory exists
    predictions_dir = f'/data2/logs/{file_name}' 
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Directory {predictions_dir} does not exist. Please check the file name.")
    # Check if the directory is empty
    if not os.listdir(predictions_dir):
        raise ValueError(f"No prediction files found in {predictions_dir}. Please ensure the files are present.")
    # Load predictions from CSV files
    galaxiesml_predictions = pd.read_csv('your_path/galaxiesml_predictions.csv')
    transferz_predictions = pd.read_csv('/your_path/{file_name}/transferz_predictions.csv')
    combo_predictions = pd.read_csv('/your_path/{file_name}/combo_predictions.csv')
    return galaxiesml_predictions, transferz_predictions, combo_predictions

def compute_lsst_metrics(truth, predictions):
    metrics = get_point_metrics(pd.Series(predictions),
                                pd.Series(truth),
                                binrange=np.linspace(0.3, 1.5, 2))
    return float(metrics['scatter_conv']), float(metrics['bias_conv']), float(metrics['outlier_conv']), float(metrics['catastrophic_outliers']) # Ensure scalars

def plot_metrics_row(model_names, model_ids, filename):
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Metrics storage
    scatter_zspec, bias_zspec, catout_zspec = [], [], []
    scatter_zphot, bias_zphot, catout_zphot = [], [], []
    scatter_combo, bias_combo, catout_combo = [], [], []

    # Compute metrics
    for model_id in model_ids:
        galaxiesml_preds, transferz_preds, combo_preds = get_predictions(model_id)

        # GalaxiesML (z_spec)
        sc1, b1, _, co1 = compute_lsst_metrics(
            galaxiesml_preds["Ground Truth"].values,
            galaxiesml_preds["Predictions"].values
        )

        # TransferZ (z_photo)
        sc2, b2, _, co2 = compute_lsst_metrics(
            transferz_preds["Ground Truth"].values,
            transferz_preds["Predictions"].values
        )

        # Combo (z_spec + z_photo)
        sc3, b3, _, co3 = compute_lsst_metrics(
            combo_preds["Ground Truth"].values,
            combo_preds["Predictions"].values
        )

        scatter_zspec.append(sc1); bias_zspec.append(b1); catout_zspec.append(co1)
        scatter_zphot.append(sc2); bias_zphot.append(b2); catout_zphot.append(co2)
        scatter_combo.append(sc3); bias_combo.append(b3); catout_combo.append(co3)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
    x = np.arange(len(model_names))
    width = 0.25  # Adjusted for 3 bars

    metric_sets = [
        ("Bias", bias_zspec, bias_zphot, bias_combo, axes[0], [-0.003, 0.003]),
        ("Scatter", scatter_zspec, scatter_zphot, scatter_combo, axes[1], [0, 0.02]),
        ("Catastrophic \n Outlier Rate", catout_zspec, catout_zphot, catout_combo, axes[2], [0, 0.1])
    ]

    colors = {
        "z_spec": "tab:blue",
        "z_photo": "tab:red",
        "combo": "xkcd:purple"
    }

    for label, vals_spec, vals_phot, vals_combo, ax, lsst_band in metric_sets:
        ax.bar(x - width, vals_phot, width, label=r"TransferZ-Images", color=colors["z_photo"])
        ax.bar(x, vals_spec, width, label=r"GalaxiesML", color=colors["z_spec"])
        ax.bar(x + width, vals_combo, width, label=r"Combo", color=colors["combo"])

        ax.set_ylabel(label, fontsize=30)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Set custom ylim
        if label == "Bias":
            ax.set_ylim(-0.01, 0.01)
        elif label == "Scatter":
            ax.set_ylim(0, 0.06)

        # LSST shaded region (skip cat outliers if needed)
        if label != "Catastrophic \n Outlier Rate":
            ax.axhspan(lsst_band[0], lsst_band[1], alpha=0.2, facecolor='green',
                       linestyle='--', linewidth=2, edgecolor='black')

        # Best metric star
        for vals, offset in zip([vals_phot, vals_spec, vals_combo], [-width, 0, width]):
            best_idx = np.argmin(np.abs(vals))
            ax.text(
                x[best_idx] + offset,
                vals[best_idx] + 0.002,
                "★", color='black', fontsize=22, fontweight='bold',
                ha='center', va='bottom'
            )
        # Arrow for bars that exceed plot top
# Arrow for bars that exceed the top of the axis
        ymin, ymax = ax.get_ylim()
        for vals, offset in zip([vals_phot, vals_spec, vals_combo], [-width, 0, width]):
            for i, v in enumerate(vals):
                if v > ymax * 0.98:
                    arrow_y = ymax - 0.07 * (ymax - ymin)  # 5% below top of axis
                    ax.text(
                        x[i] + offset,
                        arrow_y,
                        "↑",  # or u'\u25B2' for ▲
                        color='black',
                        fontsize=22,
                        ha='center',
                        va='bottom'
                    )


    # X-axis labels
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=26, rotation=30)

    # Shared legend
    lsst_patch = Line2D([], [], color='green', linewidth=10, alpha=0.2, label="LSST Requirement")
    star_handle = Line2D([], [], color='black', marker='*', linestyle='None',
                         markersize=16, label='Best Metric')
    zphot_handle = Line2D([], [], color=colors["z_photo"], linewidth=10, label=r"TransferZ-Images")
    zspec_handle = Line2D([], [], color=colors["z_spec"], linewidth=10, label=r"GalaxiesML")
    combo_handle = Line2D([], [], color=colors["combo"], linewidth=10, label=r"Combo")

    fig.legend(
        handles=[zphot_handle, zspec_handle, combo_handle], #removed lsst_handle and star_handle
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=5,
        fontsize=28
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_dir = "neuripsml4ps2025/plots"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{filename}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/{filename}.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_metrics_vs_fraction(frac_models, filename="metrics_vs_fraction"):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Parse training fractions (e.g., 10, 20, ..., 100)
    fractions = [int(name.split("_")[-1]) for name in frac_models]

    plt.rcParams.update({
        "axes.labelsize": 36,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    })

    # Storage
    scatter_zspec, bias_zspec, catout_zspec = [], [], []
    scatter_combo, bias_combo, catout_combo = [], [], []
    scatter_zphot, bias_zphot, catout_zphot = [], [], []

    # ---- Include TransferZ_Baseline at fraction = 0 ----
    baseline_model_id = "TransferZ_Baseline"
    galaxiesml_preds, transferz_preds, combo_preds = get_predictions(baseline_model_id)

    sc1, b1, _, co1 = compute_lsst_metrics(galaxiesml_preds["Ground Truth"].values, galaxiesml_preds["Predictions"].values)
    sc2, b2, _, co2 = compute_lsst_metrics(combo_preds["Ground Truth"].values, combo_preds["Predictions"].values)
    sc3, b3, _, co3 = compute_lsst_metrics(transferz_preds["Ground Truth"].values, transferz_preds["Predictions"].values)

    scatter_zspec.append(sc1); bias_zspec.append(b1); catout_zspec.append(co1)
    scatter_combo.append(sc2); bias_combo.append(b2); catout_combo.append(co2)
    scatter_zphot.append(sc3); bias_zphot.append(b3); catout_zphot.append(co3)

    fractions = [0] + fractions  # Insert fraction=0 at front

    # ---- Loop over fractional models ----
    for model_name in frac_models:
        galaxiesml_preds, transferz_preds, combo_preds = get_predictions(model_name)

        sc1, b1, _, co1 = compute_lsst_metrics(galaxiesml_preds["Ground Truth"].values, galaxiesml_preds["Predictions"].values)
        sc2, b2, _, co2 = compute_lsst_metrics(combo_preds["Ground Truth"].values, combo_preds["Predictions"].values)
        sc3, b3, _, co3 = compute_lsst_metrics(transferz_preds["Ground Truth"].values, transferz_preds["Predictions"].values)

        scatter_zspec.append(sc1); bias_zspec.append(b1); catout_zspec.append(co1)
        scatter_combo.append(sc2); bias_combo.append(b2); catout_combo.append(co2)
        scatter_zphot.append(sc3); bias_zphot.append(b3); catout_zphot.append(co3)

    # Setup 1x3 plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True)

    colors = {
        "z_spec": "tab:blue",
        "combo": "xkcd:purple",
        "z_photo": "tab:red"
    }

    # ---- Bias ----
    ax = axes[0]
    ax.plot(fractions, bias_zspec, marker='o', linewidth=5, color=colors["z_spec"], label="GalaxiesML")
    ax.plot(fractions, bias_combo, marker='o', linewidth=5, color=colors["combo"], label="Combo")
    ax.plot(fractions, bias_zphot, marker='o', linewidth=5, color=colors["z_photo"], label="TransferZ-Images")
    ax.axhspan(-0.003, 0.003, color="green", alpha=0.2) #label="LSST Requirement"
    ax.set_ylabel("Bias")
    ax.set_xlabel("Training Fraction (%)")
    ax.grid(True)

    # ---- Scatter ----
    ax = axes[1]
    ax.plot(fractions, scatter_zspec, marker='o', linewidth=5, color=colors["z_spec"])
    ax.plot(fractions, scatter_combo, marker='o', linewidth=5, color=colors["combo"])
    ax.plot(fractions, scatter_zphot, marker='o', linewidth=5, color=colors["z_photo"])
    ax.axhspan(0, 0.02, color="green", alpha=0.2)
    ax.set_ylabel("Scatter")
    ax.set_xlabel("Training Fraction (%)")
    ax.grid(True)

    # ---- Catastrophic Outlier Rate ----
    ax = axes[2]
    ax.plot(fractions, catout_zspec, marker='o', linewidth=5, color=colors["z_spec"])
    ax.plot(fractions, catout_combo, marker='o', linewidth=5, color=colors["combo"])
    ax.plot(fractions, catout_zphot, marker='o', linewidth=5, color=colors["z_photo"])
    # ax.axhspan(0, 0.1, color="green", alpha=0.2)
    ax.set_ylabel("Catastrophic \n Outlier Rate")
    ax.set_xlabel("Training Fraction (%)")
    ax.grid(True)

    # Shared Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=36)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_dir = "neuripsml4ps2025/plots"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{filename}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_dir}/{filename}.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # Creates metrics table
    model_row_names = ['CNN-Base', 'CNN-TL', 'CNN-LoRA', 'CNN-Combo', 'CNN-Base-Rev', 'CNN-TL-Rev', 'CNN-LoRA-Rev']
    model_row_ids = [
        'TransferZ_Baseline',
        'TZ_to_GM_Standard_TL',
        'TZ_to_GM_LoRA_Full',
        'Combo_Model',
        'GalaxiesML_Baseline',
        'GM_to_TZ_Standard_TL',
        'GM_to_TZ_LoRA_Full'
    ]
    make_metrics_table(model_row_names, model_row_ids, 'metrics_table')
    # Creates Figure 3
    plot_metrics_row(model_row_names, model_row_ids, 'metric_row')
    # Creates Figure 4
    frac_models = ['TZbase_GMLoraFull_frac_10', 
                    'TZbase_GMLoraFull_frac_20', 
                    'TZbase_GMLoraFull_frac_30', 
                    'TZbase_GMLoraFull_frac_40', 
                    'TZbase_GMLoraFull_frac_50',
                    'TZbase_GMLoraFull_frac_60',
                    'TZbase_GMLoraFull_frac_70',
                    'TZbase_GMLoraFull_frac_80',
                    'TZbase_GMLoraFull_frac_90',
                    'TZbase_GMLoraFull_frac_100']
    plot_metrics_vs_fraction(frac_models, filename="metrics_vs_fraction_TZtoGM")
    # Creates Figure 5
    model_reverse_comparison_names = ['CNN-Base', 'CNN-Base-Rev', 'CNN-LoRA', 'CNN-LoRA-Rev']
    model_reverse_comparison_ids = [
        'TransferZ_Baseline',
        'GalaxiesML_Baseline',
        'TZ_to_GM_LoRA_Full',
        'GM_to_TZ_LoRA_Full'
    ]
    plot_metrics_row(model_reverse_comparison_names, model_reverse_comparison_ids, 'metric_row_reverse_comparison')


