import os
import mmcv
import argparse
import numpy as np
import seaborn as sns
import matplotlib
import pathlib

matplotlib.use('Agg')  # Use Agg backend for generating plots without a display
import matplotlib.pyplot as plt


def plot_heatmap(xy_coords, save_path='heatmap_output.jpg', is_box_dim=False):
    """
    Generate a heatmap visualization of 2D coordinate distributions with marginal histograms.

    Args:
        xy_coords: Numpy array of shape (N, 2) containing x,y coordinates
        save_path: Path where the visualization image will be saved
        is_box_dim: Boolean indicating if we're plotting box dimensions (width/height)
    """
    xy_coords = np.asarray(xy_coords)
    assert (
        xy_coords.ndim == 2 and xy_coords.shape[1] == 2
    ), "xy_coords should be a 2D array with shape (N, 2)"

    x = xy_coords[:, 0]
    y = xy_coords[:, 1]

    # Create figure with specific layout for main heatmap and marginal histograms
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(4, 4, hspace=0.6, wspace=0.6)

    ax_main = fig.add_subplot(grid[1:, :-1])  # Main heatmap in center
    ax_xhist = fig.add_subplot(grid[0, :-1])  # X histogram on top
    ax_yhist = fig.add_subplot(grid[1:, -1])  # Y histogram on right

    # Main heatmap showing density of points
    if is_box_dim:
        # For box dimensions, use fewer levels and adjust parameters to avoid contour level errors
        sns.kdeplot(
            x=x, y=y, fill=True, cmap="Blues", thresh=0.05, levels=30, ax=ax_main
        )
        ax_main.set_xlim(0, np.max(x) * 1.1)
        ax_main.set_ylim(0, np.max(y) * 1.1)
        ax_main.set_xlabel("Width (pixels)")
        ax_main.set_ylabel("Height (pixels)")
        ax_main.set_title("2D Bounding Box Dimensions Heatmap")
    else:
        # For position coordinates, use original parameters
        sns.kdeplot(
            x=x, y=y, fill=True, cmap="Blues", thresh=0.01, levels=100, ax=ax_main
        )
        ax_main.set_xlim(0, 1920)
        ax_main.set_ylim(0, 1080)
        ax_main.invert_yaxis()  # Invert Y-axis to match image coordinates (0 at top)
        ax_main.set_xlabel("U-axis (pixels)")
        ax_main.set_ylabel("V-axis (pixels)")
        ax_main.set_title("2D Bounding Box Center Heatmap")

    # X-axis distribution histogram on top
    ax_xhist.hist(x, bins=100, color='blue', alpha=0.6)
    ax_xhist.set_xlim(ax_main.get_xlim())
    ax_xhist.set_title(f"{'Width' if is_box_dim else 'U-axis'} Distribution")
    ax_xhist.axis('off')

    # Y-axis distribution histogram on right
    ax_yhist.hist(y, bins=100, orientation='horizontal', color='blue', alpha=0.6)
    ax_yhist.set_ylim(ax_main.get_ylim())
    if not is_box_dim:
        ax_yhist.invert_yaxis()
    ax_yhist.set_title(f"{'Height' if is_box_dim else 'V-axis'} Distribution")
    ax_yhist.axis('off')

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(
        f"{'Box dimension' if is_box_dim else 'Box center'} distribution image has been saved to: {save_path}"
    )


def get_path_components(pkl_path):
    """
    Extract parent and parent-of-parent directory names from path.

    Args:
        pkl_path: Path to the pickle file

    Returns:
        A string with format "parent_of_parent_folder_name_parent_folder_name"
    """
    path = pathlib.Path(pkl_path)
    # Get parent folder (immediate directory containing the file)
    parent = path.parent.name
    # Get parent of parent folder (directory containing the parent folder)
    parent_of_parent = path.parent.parent.name if path.parent.parent else ""

    if parent_of_parent:
        return f"{parent_of_parent}_{parent}"
    return parent


def visualize_2d_box_distribution(
    info, frame_num, output_dir="result_vis/anchors", path_suffix=""
):
    """
    Visualize the distribution of 2D bounding box centers across multiple frames.

    Args:
        info: List of dictionaries containing frame information with bounding boxes
        frame_num: Number of frames to process
        output_dir: Directory where the visualization will be saved
        path_suffix: Suffix to add to output filename based on input path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect center points, box nums, and box dimensions of all 2D bounding boxes
    coords = []
    box_nums = []
    for i in range(frame_num):
        for gt_boxs_cam in info[i]['bboxes2d']:
            box_nums.append(len(gt_boxs_cam))
            for gtbox in gt_boxs_cam:
                x1, y1, x2, y2 = gtbox[:4].astype(int)
                cx = (x1 + x2) / 2  # Calculate center x-coordinate
                cy = (y1 + y2) / 2  # Calculate center y-coordinate
                w = x2 - x1
                h = y2 - y1
                coords.append([cx, cy, w, h])
    coords = np.array(coords)
    box_nums = np.array(box_nums)

    # Print statistics about coordinates range
    print(
        f"U-axis range: {np.min(coords[:, 0])} to {np.max(coords[:, 0])}, mean: {np.mean(coords[:, 0])}, median: {np.median(coords[:, 0])}"
    )
    print(
        f"V-axis range: {np.min(coords[:, 1])} to {np.max(coords[:, 1])}, mean: {np.mean(coords[:, 1])}, median: {np.median(coords[:, 1])}"
    )
    print(
        f"Box width range: {np.min(coords[:, 2])} to {np.max(coords[:, 2])}, mean: {np.mean(coords[:, 2])}, median: {np.median(coords[:, 2])}"
    )
    print(
        f"Box height range: {np.min(coords[:, 3])} to {np.max(coords[:, 3])}, mean: {np.mean(coords[:, 3])}, median: {np.median(coords[:, 3])}"
    )
    print(
        f"Box nums range: {np.min(box_nums)} to {np.max(box_nums)}, mean: {np.mean(box_nums)}, median: {np.median(box_nums)}"
    )

    # Generate and save the heatmap visualization with path-based filename
    box_center_filename = (
        f"2d_box_distribution_{path_suffix}.jpg"
        if path_suffix
        else "2d_box_distribution.jpg"
    )
    plot_heatmap(coords[:, :2], os.path.join(output_dir, box_center_filename))

    box_dim_filename = (
        f"2d_box_dim_distribution_{path_suffix}.jpg"
        if path_suffix
        else "2d_box_dim_distribution.jpg"
    )
    plot_heatmap(
        coords[:, 2:], os.path.join(output_dir, box_dim_filename), is_box_dim=True
    )


def visualize_depth_distribution(
    info, frame_num, output_dir="result_vis/anchors", path_suffix=""
):
    """
    Visualize the distribution of 2D bounding box depths across multiple frames.

    Args:
        info: List of dictionaries containing frame information with depths
        frame_num: Number of frames to process
        output_dir: Directory where the visualization will be saved
        path_suffix: Suffix to add to output filename based on input path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect depths of all 2D bounding boxes
    depths = []
    for i in range(frame_num):
        for depth in info[i]['depths']:
            depths.extend(depth)

    depths = np.array(depths)

    print(
        f"Depth range: {np.min(depths)} to {np.max(depths)}, mean: {np.mean(depths)}, median: {np.median(depths)}"
    )

    # Generate and save the depth distribution visualization with histogram
    plt.figure(figsize=(12, 8))

    # Create histogram with KDE
    sns.histplot(depths, kde=True, bins=100, color='blue')

    # Add vertical lines for statistics
    plt.axvline(
        np.mean(depths),
        color='red',
        linestyle='--',
        label=f'Mean: {np.mean(depths):.2f}',
    )
    plt.axvline(
        np.median(depths),
        color='green',
        linestyle=':',
        label=f'Median: {np.median(depths):.2f}',
    )

    plt.title('Ground Truth Depth Distribution')
    plt.xlabel('Depth (m)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)

    # Save the figure with path-based filename
    filename = (
        f"depth_distribution_{path_suffix}.jpg"
        if path_suffix
        else "depth_distribution.jpg"
    )
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Depth distribution image has been saved to: {save_path}")


def main():
    """
    Main function to parse arguments and run the visualization.
    """
    parser = argparse.ArgumentParser(
        description="Visualize the distribution of 2D bounding boxes"
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="./data/infos/V2X-Seq-SPD-2hz/infrastructure-side/modify_infos_train.pkl",
        help="Path to the pickle file containing dataset information",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="Number of frames to visualize (use 0 for all frames)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result_vis/anchors",
        help="Output directory for the visualized images",
    )
    args = parser.parse_args()

    # Load the dataset information
    data = mmcv.load(args.pkl_path)

    # Determine how many frames to process (all frames if num_frames=0)
    visual_frame_num = (
        min(args.num_frames, len(data['infos']))
        if args.num_frames > 0
        else len(data['infos'])
    )

    # Extract path components to use as filename suffix
    path_suffix = get_path_components(args.pkl_path)

    # Generate the 2d box center distribution visualization
    visualize_2d_box_distribution(
        data['infos'], visual_frame_num, args.output_dir, path_suffix
    )
    print("-" * 50)
    # Generate the depth distribution visualization
    visualize_depth_distribution(
        data['infos'], visual_frame_num, args.output_dir, path_suffix
    )


if __name__ == "__main__":
    main()
