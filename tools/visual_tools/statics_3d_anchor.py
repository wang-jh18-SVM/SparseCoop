import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize anchor points from .npy file"
    )
    parser.add_argument(
        "--anchor_file",
        type=str,
        default="data/infos/V2X-Seq-SPD-2hz-for-SparseCoop/vehicle-side/spd_kmeans900.npy",
        help="Path to the anchor file (.npy)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="result_vis/anchors",
        help="Directory to save visualizations",
    )
    return parser.parse_args()


def get_file_suffix(file_path):
    """Extract a meaningful suffix from the file path for naming"""
    path = pathlib.Path(file_path)
    # Return the stem (filename without extension) and its parent folder (if exists) as the suffix
    parent_name = path.parent.name
    if parent_name:
        return f"{parent_name}_{path.stem}"
    return path.stem


def visualize_anchors(anchor_file, save_dir="result_vis/anchors"):
    """
    Visualize the loaded anchor points with histograms and 2D projections

    Args:
        anchor_file: Path to the anchor file (.npy)
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load anchors
    anchors = np.load(anchor_file)
    anchors_x = anchors[:, 0]
    anchors_y = anchors[:, 1]
    anchors_z = anchors[:, 2]

    # Get suffix for naming
    suffix = get_file_suffix(anchor_file)

    # Create distribution histograms
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    # X coordinate distribution
    axes[0].hist(anchors_x, bins=50, color='blue', alpha=0.7)
    axes[0].set_title(f'X Distribution of Anchors ({suffix})', fontsize=14)
    axes[0].grid(True)

    # Y coordinate distribution
    axes[1].hist(anchors_y, bins=50, color='blue', alpha=0.7)
    axes[1].set_title(f'Y Distribution of Anchors ({suffix})', fontsize=14)
    axes[1].grid(True)

    # Z coordinate distribution
    axes[2].hist(anchors_z, bins=50, color='blue', alpha=0.7)
    axes[2].set_title(f'Z Distribution of Anchors ({suffix})', fontsize=14)
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/anchor_distribution_{suffix}.png', dpi=300)
    plt.close()

    # 2D projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # X-Y projection
    axes[0].scatter(anchors_x, anchors_y, c='blue', marker='o', alpha=0.5)
    axes[0].set_title(f'X-Y Projection ({suffix})', fontsize=14)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True)

    # X-Z projection
    axes[1].scatter(anchors_x, anchors_z, c='blue', marker='o', alpha=0.5)
    axes[1].set_title(f'X-Z Projection ({suffix})', fontsize=14)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].grid(True)

    # Y-Z projection
    axes[2].scatter(anchors_y, anchors_z, c='blue', marker='o', alpha=0.5)
    axes[2].set_title(f'Y-Z Projection ({suffix})', fontsize=14)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/anchor_2d_projections_{suffix}.png', dpi=300)
    plt.close()

    # # 3D visualization
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(anchors_x, anchors_y, anchors_z, c='blue', marker='o', alpha=0.6)
    # ax.set_title(f'3D Visualization of Anchor Points ({suffix})', fontsize=16)
    # ax.set_xlabel('X', fontsize=14)
    # ax.set_ylabel('Y', fontsize=14)
    # ax.set_zlabel('Z', fontsize=14)
    # ax.grid(True)

    # # Set appropriate view
    # ax.view_init(elev=30, azim=45)
    # plt.tight_layout()
    # plt.savefig(f'{save_dir}/anchor_3d_{suffix}.png', dpi=300, bbox_inches='tight')
    # plt.close()


if __name__ == "__main__":
    args = parse_args()
    visualize_anchors(args.anchor_file, args.save_dir)
