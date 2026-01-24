import os
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


def convert_pcd_to_bin(pcd_file_path, bin_file_path):
    """
    Convert a PCD file to BIN format.

    Args:
        pcd_file_path (str): Path to the input PCD file
        bin_file_path (str): Path to the output BIN file
    """
    try:
        # Read PCD file using Open3D
        pcd = o3d.io.read_point_cloud(pcd_file_path)

        # Get points as numpy array
        points = np.asarray(pcd.points)
        # Reshape to [N, 3]
        point_data = points.reshape(-1, 3)

        # Save as binary file (float32 format is commonly used for point clouds)
        point_data = point_data.astype(np.float32)
        point_data.tofile(bin_file_path)

        # print(
        #     f"Successfully converted: {os.path.basename(pcd_file_path)} -> {os.path.basename(bin_file_path)}"
        # )
        return True

    except Exception as e:
        print(f"Error converting {pcd_file_path}: {str(e)}")
        return False


def compare_pcd_and_bin(
    pcd_file_path, bin_file_path, save_folder="result_vis/debug_pcd_bin_compare"
):
    """
    Compare PCD and BIN files by visualizing them side by side and checking data consistency.

    Args:
        pcd_file_path (str): Path to the original PCD file
        bin_file_path (str): Path to the converted BIN file
    """
    print(
        f"\n=== Comparing {os.path.basename(pcd_file_path)} and {os.path.basename(bin_file_path)} ==="
    )

    try:
        # Read PCD file
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        pcd_points = np.asarray(pcd.points)

        # Read BIN file
        bin_points = np.fromfile(bin_file_path, dtype=np.float32)
        bin_points = bin_points.reshape(-1, 3)

        # Check data consistency
        print(f"PCD points shape: {pcd_points.shape}")
        print(f"BIN points shape: {bin_points.shape}")

        if pcd_points.shape == bin_points.shape:
            # Calculate differences
            diff = np.abs(pcd_points - bin_points)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"Maximum difference: {max_diff:.8f}")
            print(f"Mean difference: {mean_diff:.8f}")

            if max_diff < 1e-6:
                print("✅ Conversion successful - data matches!")
            else:
                print("⚠️  Warning - significant differences detected!")
        else:
            print("❌ Error - point cloud shapes don't match!")
            return False

        # Create matplotlib subplot for comparison
        fig = plt.figure(figsize=(10, 5))

        # Plot original PCD
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(
            pcd_points[::10, 0],
            pcd_points[::10, 1],
            pcd_points[::10, 2],
            c=pcd_points[::10, 2],
            cmap='viridis',
            s=1,
        )
        ax1.set_title('Original PCD')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Plot converted BIN
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(
            bin_points[::10, 0],
            bin_points[::10, 1],
            bin_points[::10, 2],
            c=bin_points[::10, 2],
            cmap='viridis',
            s=1,
        )
        ax2.set_title('Converted BIN')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # Set BEV view
        ax1.view_init(elev=90, azim=-90)
        ax2.view_init(elev=90, azim=-90)

        plt.tight_layout()

        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(
            save_folder,
            os.path.basename(pcd_file_path).replace(".pcd", "_bin_compare.png"),
        )
        plt.savefig(save_path)
        print(f"Saved comparison figure to {save_path}")

        return True

    except Exception as e:
        print(f"Error comparing files: {str(e)}")
        return False


def visualize_sample_conversion(target_folder, sample_file=None):
    """
    Convert and visualize a sample file to check the conversion process.

    Args:
        sample_file (str): Path to a specific PCD file to test. If None, uses the first PCD file found.
    """
    if sample_file is None:
        # Find the first PCD file in the target folder
        pcd_pattern = os.path.join(target_folder, "*.pcd")
        pcd_files = glob.glob(pcd_pattern)

        if not pcd_files:
            print(f"No PCD files found in '{target_folder}'")
            return

        sample_file = pcd_files[0]

    print(
        f"\n=== Testing conversion with sample file: {os.path.basename(sample_file)} ==="
    )

    # Generate BIN file path
    file_prefix = os.path.splitext(os.path.basename(sample_file))[0]
    bin_file = os.path.join(target_folder, file_prefix + "_test.bin")

    # Convert the file
    success = convert_pcd_to_bin(sample_file, bin_file)

    if success:
        print("Comparing PCD and BIN files...")
        compare_pcd_and_bin(sample_file, bin_file)

    # Clean up test file
    if os.path.exists(bin_file):
        os.remove(bin_file)
        print(f"\nCleaned up test file: {os.path.basename(bin_file)}")


def main():
    """
    Main function to convert all PCD files in the target folder to BIN format.
    """
    import argparse

    # Add command line arguments
    parser = argparse.ArgumentParser(
        description='Convert PCD files to BIN format with optional visualization'
    )
    parser.add_argument(
        '--target-folder',
        type=str,
        default="datasets/V2X-Seq-SPD-2hz/vehicle-side/velodyne",
        help='Target folder containing PCD files',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test conversion with a sample file and show visualizations',
    )
    parser.add_argument(
        '--test_frame', type=str, default=None, help='Test frame number'
    )

    args = parser.parse_args()

    target_folder = args.target_folder

    # Check if target folder exists
    if not os.path.exists(target_folder):
        print(f"Error: Target folder '{target_folder}' does not exist!")
        return

    # Find all PCD files in the target folder
    pcd_pattern = os.path.join(target_folder, "*.pcd")
    pcd_files = glob.glob(pcd_pattern)

    if not pcd_files:
        print(f"No PCD files found in '{target_folder}'")
        return

    # If test mode is enabled, get a random sample file
    if args.test:
        if args.test_frame is not None:
            sample_file = os.path.join(target_folder, f"{args.test_frame}.pcd")
        else:
            sample_file = random.choice(pcd_files)
        visualize_sample_conversion(target_folder, sample_file)
        return

    print(f"Found {len(pcd_files)} PCD files to convert")

    # Convert each PCD file to BIN
    successful_conversions = 0
    failed_conversions = 0

    for pcd_file in tqdm(pcd_files, desc="Converting PCD to BIN"):
        # Generate BIN file path with same prefix name
        file_prefix = os.path.splitext(os.path.basename(pcd_file))[0]
        bin_file = os.path.join(target_folder, file_prefix + ".bin")

        # Convert the file
        if convert_pcd_to_bin(pcd_file, bin_file):
            successful_conversions += 1
        else:
            failed_conversions += 1

    # Print summary
    print(f"\nConversion completed!")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Total files processed: {len(pcd_files)}")


if __name__ == "__main__":
    main()
