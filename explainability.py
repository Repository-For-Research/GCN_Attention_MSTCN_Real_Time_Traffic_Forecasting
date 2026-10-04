"""
Explainability analysis for GCN-Attn-MSTCN.

This script provides three explainability analyses:

1. Spatial Attention
   - Raw attention scores before softmax
   - Softmax attention weights after softmax
   - Top-k attended neighbours

2. Peak vs. Non-Peak Spatial Attention
   - Automatically detects peak and non-peak periods
   - Compares spatial attention for a target sensor
   - Visualizes important sensors on the Los Angeles sensor map

3. Temporal Importance
   - Gradient-based temporal importance
   - Compares historical timestep importance during peak and non-peak periods


"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gcn_attn_mstcn.data import (
    build_dataloaders,
    load_adjacency,
    normalize_adj,
)
from gcn_attn_mstcn.models import GCNAttnMSTCN
from gcn_attn_mstcn.utils import (
    ensure_dir,
    get_device,
    load_config,
    set_seed,
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_model(cfg, bundle, checkpoint, device):
    """Load a trained GCNAttnMSTCN model."""

    m = cfg["model"]

    model = GCNAttnMSTCN(
        bundle.num_features,
        m["hidden"],
        m["tcn_layers"],
        cfg["data"]["output_len"],
        m["dropout"],
        m["num_heads"],
    ).to(device)

    state_dict = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_graph(cfg, device):
    """Load and normalize the graph adjacency matrix."""

    sensor_ids, _, adj_np = load_adjacency(
        cfg["data"]["adj_path"]
    )

    adj = normalize_adj(
        torch.tensor(
            adj_np,
            dtype=torch.float32,
            device=device,
        )
    )

    return sensor_ids, adj


# ---------------------------------------------------------------------
# 1. Spatial attention
# ---------------------------------------------------------------------

def spatial_attention_analysis(
    model,
    bundle,
    adj,
    sensor_ids,
    device,
    node,
    sample,
    topk,
    output_dir,
):
    """
    Analyse spatial attention for one target sensor.

    Produces:

    - Raw attention scores before softmax
    - Softmax attention weights
    - Top-k attended neighbours
    - CSV containing top-k neighbours
    """

    print("\n" + "=" * 70)
    print("1. SPATIAL ATTENTION ANALYSIS")
    print("=" * 70)

    x = (
        bundle.test.dataset[sample][0]
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        # Use the final observed timestep.
        h = model.gcn(
            x[:, -1],
            adj,
        )

        # Raw scores before softmax.
        raw = model.spatial_att(
            h,
            adj,
            return_scores=True,
        )

        # Attention weights after softmax.
        _, attention = model.spatial_att(
            h,
            adj,
            return_attention=True,
        )

    # -------------------------------------------------------------
    # Shape:
    #
    # raw      = (1, heads, N, N)
    # attention = (1, heads, N, N)
    #
    # Average across attention heads.
    # -------------------------------------------------------------

    raw = (
        raw.mean(1)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    attention = (
        attention.mean(1)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    if node < 0 or node >= raw.shape[0]:
        raise ValueError(
            f"Invalid node {node}. "
            f"Number of sensors: {raw.shape[0]}"
        )

    raw_node = raw[node]
    attention_node = attention[node]

    # -------------------------------------------------------------
    # Figure 1:
    # Raw scores vs softmax attention weights
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
    )

    axes[0].bar(
        np.arange(len(raw_node)),
        raw_node,
    )

    axes[0].set_title(
        f"Raw Spatial Attention Scores "
        f"(Sensor {node})"
    )

    axes[0].set_ylabel(
        "Raw Score"
    )

    axes[1].bar(
        np.arange(len(attention_node)),
        attention_node,
    )

    axes[1].set_title(
        "Softmax Spatial Attention Weights"
    )

    axes[1].set_xlabel(
        "Source Sensor Index"
    )

    axes[1].set_ylabel(
        "Attention Weight"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"node{node}_raw_vs_softmax.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Top-k neighbours
    # -------------------------------------------------------------

    weights = attention_node.copy()

    # Exclude self-loop.
    weights[node] = 0.0

    top_indices = np.argsort(weights)[-topk:][::-1]

    top_scores = weights[top_indices]

    top_sensor_ids = [
        sensor_ids[i]
        for i in top_indices
    ]

    top_df = pd.DataFrame(
        {
            "sensor_index": top_indices,
            "sensor_id": top_sensor_ids,
            "attention": top_scores,
        }
    )

    csv_path = (
        output_dir
        / f"node{node}_top{topk}_neighbours.csv"
    )

    top_df.to_csv(
        csv_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Top-k bar plot
    # -------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        [
            str(sensor_ids[i])
            for i in top_indices
        ][::-1],
        top_scores[::-1],
    )

    ax.set_xlabel(
        "Attention Weight"
    )

    ax.set_ylabel(
        "Sensor ID"
    )

    ax.set_title(
        f"Top-{topk} Attended Neighbours "
        f"of Sensor {node}"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"node{node}_top{topk}_neighbours.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved spatial attention results to: "
        f"{output_dir}"
    )

    return {
        "raw": raw,
        "attention": attention,
        "top_indices": top_indices,
        "top_scores": top_scores,
    }


# ---------------------------------------------------------------------
# 2. Peak / non-peak detection
# ---------------------------------------------------------------------

def detect_peak_nonpeak(
    traffic_csv,
    ignore_first=0,
):
    """
    Automatically detect peak and non-peak periods.

    The standard deviation across sensors is calculated
    for every timestep.

    Higher spatial variation:
        -> peak traffic condition

    Lower spatial variation:
        -> non-peak traffic condition

    Returns:
        peak_time
        nonpeak_time
        peak_index
        nonpeak_index
    """

    print("\n" + "=" * 70)
    print("2. PEAK / NON-PEAK DETECTION")
    print("=" * 70)

    df = pd.read_csv(
        traffic_csv,
        index_col=0,
        parse_dates=True,
    )

    # Keep only numeric sensor measurements.
    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        raise ValueError(
            "No numeric traffic data found."
        )

    # Standard deviation across sensors
    # for each timestep.
    std_per_timestep = numeric_df.std(
        axis=1
    )

    if ignore_first > 0:
        valid_std = std_per_timestep.iloc[
            ignore_first:
        ]
    else:
        valid_std = std_per_timestep

    peak_time = valid_std.idxmax()
    nonpeak_time = valid_std.idxmin()

    peak_index = df.index.get_loc(
        peak_time
    )

    nonpeak_index = df.index.get_loc(
        nonpeak_time
    )

    print(
        f"Peak time:     {peak_time}"
    )

    print(
        f"Peak std:      "
        f"{valid_std.loc[peak_time]:.4f}"
    )

    print(
        f"Non-peak time: {nonpeak_time}"
    )

    print(
        f"Non-peak std:  "
        f"{valid_std.loc[nonpeak_time]:.4f}"
    )

    return (
        df,
        peak_time,
        nonpeak_time,
        peak_index,
        nonpeak_index,
    )


# ---------------------------------------------------------------------
# 3. Spatial attention for a specific sample
# ---------------------------------------------------------------------

def get_spatial_attention_for_sample(
    model,
    bundle,
    adj,
    device,
    sample,
    node,
):
    """
    Extract head-averaged spatial attention
    for a particular test sample.
    """

    x = (
        bundle.test.dataset[sample][0]
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        h = model.gcn(
            x[:, -1],
            adj,
        )

        _, attention = model.spatial_att(
            h,
            adj,
            return_attention=True,
        )

    attention = (
        attention.mean(1)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    weights = attention[node].copy()

    # Remove self-attention.
    weights[node] = 0.0

    return weights


# ---------------------------------------------------------------------
# 4. Geographical sensor map
# ---------------------------------------------------------------------

def load_sensor_locations():
    """
    Download and load METR-LA sensor coordinates.
    """

    import urllib.request

    url = (
        "https://raw.githubusercontent.com/"
        "liyaguang/DCRNN/master/data/"
        "sensor_graph/graph_sensor_locations.csv"
    )

    local_file = Path(
        "graph_sensor_locations_la.csv"
    )

    if not local_file.exists():

        print(
            "Downloading METR-LA sensor locations..."
        )

        urllib.request.urlretrieve(
            url,
            local_file,
        )

    sensors = pd.read_csv(
        local_file
    )

    return sensors


def create_sensor_geodataframe():
    """
    Convert sensor coordinates into a GeoDataFrame.
    """

    try:

        import geopandas as gpd
        from shapely.geometry import Point

    except ImportError:

        raise ImportError(
            "Geographical visualization requires "
            "geopandas and shapely."
        )

    sensors = load_sensor_locations()

    gdf = gpd.GeoDataFrame(
        sensors,
        geometry=gpd.points_from_xy(
            sensors.longitude,
            sensors.latitude,
        ),
        crs="EPSG:4326",
    )

    return gdf.to_crs(
        epsg=3857
    )


# ---------------------------------------------------------------------
# 5. Peak vs non-peak spatial visualization
# ---------------------------------------------------------------------

def plot_spatial_comparison(
    peak_weights,
    nonpeak_weights,
    target_node,
    sensor_ids,
    output_dir,
    topk=10,
):
    """
    Compare spatial importance during peak
    and non-peak conditions.
    """

    print("\nCreating spatial comparison maps...")

    gdf = create_sensor_geodataframe()

    # -------------------------------------------------------------
    # Peak top-k
    # -------------------------------------------------------------

    peak_indices = np.argsort(
        peak_weights
    )[-topk:][::-1]

    peak_scores = peak_weights[
        peak_indices
    ]

    # -------------------------------------------------------------
    # Non-peak top-k
    # -------------------------------------------------------------

    nonpeak_indices = np.argsort(
        nonpeak_weights
    )[-topk:][::-1]

    nonpeak_scores = nonpeak_weights[
        nonpeak_indices
    ]

    # -------------------------------------------------------------
    # Plotting function
    # -------------------------------------------------------------

    def make_map(
        indices,
        scores,
        title,
        filename,
    ):

        import contextily as ctx

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        # All sensors.
        gdf.plot(
            ax=ax,
            color="gray",
            alpha=0.3,
            markersize=10,
        )

        # Target sensor.
        gdf.iloc[
            [target_node]
        ].plot(
            ax=ax,
            color="red",
            markersize=80,
            label="Target sensor",
        )

        # Highlight top-k sensors.
        for rank, idx in enumerate(indices):

            color = (
                "darkblue"
                if rank >= len(indices) // 2
                else "lightblue"
            )

            gdf.iloc[
                [idx]
            ].plot(
                ax=ax,
                color=color,
                markersize=35,
            )

            x = gdf.iloc[
                idx
            ].geometry.x

            y = gdf.iloc[
                idx
            ].geometry.y

            ax.text(
                x - 0.0008,
                y,
                f"{scores[rank]:.3f}",
                fontsize=7,
                color="black",
                fontweight="bold",
            )

        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
        )

        ax.set_title(
            title
        )

        ax.axis("off")

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            output_dir / filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    # Peak map.
    make_map(
        peak_indices,
        peak_scores,
        f"Spatial Explainability "
        f"for Sensor {target_node} "
        f"(Peak)",
        f"node{target_node}_spatial_peak.png",
    )

    # Non-peak map.
    make_map(
        nonpeak_indices,
        nonpeak_scores,
        f"Spatial Explainability "
        f"for Sensor {target_node} "
        f"(Non-Peak)",
        f"node{target_node}_spatial_nonpeak.png",
    )

    # -------------------------------------------------------------
    # Save comparison table.
    # -------------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "peak_sensor_index": peak_indices,
            "peak_sensor_id": [
                sensor_ids[i]
                for i in peak_indices
            ],
            "peak_attention": peak_scores,
            "nonpeak_sensor_index": nonpeak_indices,
            "nonpeak_sensor_id": [
                sensor_ids[i]
                for i in nonpeak_indices
            ],
            "nonpeak_attention": nonpeak_scores,
        }
    )

    comparison.to_csv(
        output_dir
        / f"node{target_node}_peak_vs_nonpeak.csv",
        index=False,
    )

    return (
        peak_indices,
        peak_scores,
        nonpeak_indices,
        nonpeak_scores,
    )


# ---------------------------------------------------------------------
# 6. Temporal importance
# ---------------------------------------------------------------------

def temporal_importance(
    model,
    bundle,
    adj,
    device,
    sample,
    target_node,
):
    """
    Calculate gradient-based temporal importance.

    The importance of each historical timestep is measured
    using the absolute gradient of the target prediction
    with respect to the input at that timestep.
    """

    x = (
        bundle.test.dataset[sample][0]
        .unsqueeze(0)
        .to(device)
    )

    x.requires_grad_(True)

    model.zero_grad()

    prediction = model(
        x,
        adj,
    )

    # First output horizon, target sensor.
    target_prediction = prediction[
        0,
        0,
        target_node,
    ]

    target_prediction.backward()

    # Assumes feature 0 is the traffic feature.
    importance = (
        x.grad.abs()
        [0, :, target_node, 0]
        .detach()
        .cpu()
        .numpy()
    )

    return importance


def plot_temporal_comparison(
    peak_importance,
    nonpeak_importance,
    output_dir,
    peak_label,
    nonpeak_label,
):
    """
    Plot temporal importance for peak and non-peak periods.
    """

    timesteps = np.arange(
        len(peak_importance)
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        timesteps,
        peak_importance,
        label=f"Peak ({peak_label})",
        linewidth=2,
    )

    ax.plot(
        timesteps,
        nonpeak_importance,
        label=f"Non-Peak ({nonpeak_label})",
        linewidth=2,
    )

    ax.set_xlabel(
        "Historical Timesteps"
    )

    ax.set_ylabel(
        "Gradient-based Importance"
    )

    ax.set_title(
        "Temporal Importance Comparison"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "temporal_importance_peak_vs_nonpeak.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Save numerical values.
    temporal_df = pd.DataFrame(
        {
            "timestep": timesteps,
            "peak_importance": peak_importance,
            "nonpeak_importance": nonpeak_importance,
        }
    )

    temporal_df.to_csv(
        output_dir
        / "temporal_importance.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Explainability analysis for "
            "GCN-Attn-MSTCN."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/metr_la.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--node",
        type=int,
        default=101,
        help="Target sensor index.",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Test sample index.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--traffic-csv",
        type=str,
        default=None,
        help=(
            "Traffic CSV used for automatic "
            "peak/non-peak detection."
        ),
    )

    parser.add_argument(
        "--peak-sample",
        type=int,
        default=None,
        help=(
            "Test sample to use for peak "
            "spatial/temporal analysis."
        ),
    )

    parser.add_argument(
        "--nonpeak-sample",
        type=int,
        default=None,
        help=(
            "Test sample to use for non-peak "
            "spatial/temporal analysis."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "spatial",
            "peak_nonpeak",
            "temporal",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--out-dir",
        default="images/explainability",
    )

    parser.add_argument(
        "--ignore-first",
        type=int,
        default=0,
        help=(
            "Number of initial traffic timesteps "
            "to ignore during peak detection."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    cfg = load_config(
        args.config
    )

    set_seed(
        cfg["seed"]
    )

    device = get_device(
        cfg["device"]
    )

    output_dir = ensure_dir(
        args.out_dir
    )

    print(
        f"Using device: {device}"
    )

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    print("\nLoading dataset...")

    bundle = build_dataloaders(
        cfg
    )

    sensor_ids, adj = load_graph(
        cfg,
        device,
    )

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    print("\nLoading trained model...")

    model = load_model(
        cfg,
        bundle,
        args.checkpoint,
        device,
    )

    print(
        "Model loaded successfully."
    )

    # -------------------------------------------------------------
    # 1. Basic spatial attention
    # -------------------------------------------------------------

    spatial_result = None

    if args.mode in [
        "spatial",
        "all",
    ]:

        spatial_result = (
            spatial_attention_analysis(
                model=model,
                bundle=bundle,
                adj=adj,
                sensor_ids=sensor_ids,
                device=device,
                node=args.node,
                sample=args.sample,
                topk=args.topk,
                output_dir=output_dir,
            )
        )

    # -------------------------------------------------------------
    # 2. Peak vs non-peak
    # -------------------------------------------------------------

    if args.mode in [
        "peak_nonpeak",
        "all",
    ]:

        if args.traffic_csv is None:

            print(
                "\nWARNING:"
                "\n--traffic-csv was not provided."
                "\nPeak/non-peak analysis skipped."
            )

        else:

            (
                df,
                peak_time,
                nonpeak_time,
                peak_index,
                nonpeak_index,
            ) = detect_peak_nonpeak(
                args.traffic_csv,
                args.ignore_first,
            )

            print(
                "\nDetected peak period:"
            )

            print(
                f"  {peak_time}"
            )

            print(
                "\nDetected non-peak period:"
            )

            print(
                f"  {nonpeak_time}"
            )

            # -----------------------------------------------------
            # IMPORTANT:
            #
            # The automatically detected raw-data timestep is not
            # necessarily identical to a test dataset sample index.
            #
            # Therefore the user can explicitly provide:
            #
            # --peak-sample
            # --nonpeak-sample
            #
            # to ensure the correct windows are analysed.
            # -----------------------------------------------------

            if (
                args.peak_sample is None
                or args.nonpeak_sample is None
            ):

                print(
                    "\nPeak/non-peak timestamps were detected, "
                    "but --peak-sample and --nonpeak-sample "
                    "were not provided."
                )

                print(
                    "Provide the corresponding test sample "
                    "indices to generate the spatial maps."
                )

            else:

                peak_weights = (
                    get_spatial_attention_for_sample(
                        model,
                        bundle,
                        adj,
                        device,
                        args.peak_sample,
                        args.node,
                    )
                )

                nonpeak_weights = (
                    get_spatial_attention_for_sample(
                        model,
                        bundle,
                        adj,
                        device,
                        args.nonpeak_sample,
                        args.node,
                    )
                )

                plot_spatial_comparison(
                    peak_weights=peak_weights,
                    nonpeak_weights=nonpeak_weights,
                    target_node=args.node,
                    sensor_ids=sensor_ids,
                    output_dir=output_dir,
                    topk=args.topk,
                )

                # -------------------------------------------------
                # 3. Temporal importance
                # -------------------------------------------------

                if args.mode in [
                    "temporal",
                    "all",
                    "peak_nonpeak",
                ]:

                    peak_temporal = (
                        temporal_importance(
                            model=model,
                            bundle=bundle,
                            adj=adj,
                            device=device,
                            sample=args.peak_sample,
                            target_node=args.node,
                        )
                    )

                    nonpeak_temporal = (
                        temporal_importance(
                            model=model,
                            bundle=bundle,
                            adj=adj,
                            device=device,
                            sample=args.nonpeak_sample,
                            target_node=args.node,
                        )
                    )

                    plot_temporal_comparison(
                        peak_importance=peak_temporal,
                        nonpeak_importance=nonpeak_temporal,
                        output_dir=output_dir,
                        peak_label=str(
                            peak_time
                        ),
                        nonpeak_label=str(
                            nonpeak_time
                        ),
                    )

                    print(
                        "\nSaved temporal importance results."
                    )

    # -------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "Explainability analysis completed."
    )

    print(
        f"Results saved to: {output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()