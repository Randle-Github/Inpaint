import argparse
import os
import subprocess
import sys

def main():
    """
    Main function to parse arguments, set up the environment, and run the
    Depth Anything video processing script.
    """
    # --- Argument Parsing (equivalent to the bash while loop) ---
    parser = argparse.ArgumentParser(
        description="A Python launcher for the Depth Anything V2 video script. "
                    "This script handles the symbolic link workaround for model weights."
    )
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="The full path to the input video file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The full path to the directory where output will be saved."
    )
    args = parser.parse_args()

    # --- Configuration (equivalent to the bash variables) ---
    python_script_path = "/coc/flash7/yliu3735/workspace/Inpaint/depth_anything/run_video.py"
    model_weights_path = "/coc/flash7/yliu3735/workspace/Inpaint/weights/depth_anything_v2_vitl.pth"
    expected_weights_path = "checkpoints/depth_anything_v2_vitl.pth"

    # --- Execution ---
    print("🚀 Starting Depth Anything V2 process...")

    # --- WORKAROUND (equivalent to mkdir and ln -sf) ---
    print("🔧 Setting up symbolic link for model weights...")
    try:
        # Create the 'checkpoints' directory if it doesn't exist
        link_dir = os.path.dirname(expected_weights_path)
        os.makedirs(link_dir, exist_ok=True)

        # Replicate 'ln -sf': Remove the link if it exists, then create it.
        # Use os.path.lexists to check for the link itself, not its target.
        if os.path.lexists(expected_weights_path):
            os.remove(expected_weights_path)
        os.symlink(model_weights_path, expected_weights_path)
        
    except OSError as e:
        print(f"❌ Error setting up symbolic link: {e}")
        sys.exit(1)

    print("========================================")
    print(f"▶️  Input Video: {args.input_video}")
    print(f"📂 Output Directory: {args.output_dir}")
    print("🤖 Model: v2-large (vitl encoder)")
    print("========================================")

    # Create the final output directory (equivalent to mkdir -p)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Run the Python script (equivalent to the python ... command) ---
    command = [
        "python",
        python_script_path,
        "--encoder", "vitl",
        "--video-path", args.input_video,
        "--outdir", args.output_dir,
    ]
    
    try:
        # Using subprocess.run to execute the command
        subprocess.run(command, check=True)
        print(f"✅ Processing complete! Output saved in {args.output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ The processing script failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Error: Could not find the script to run at '{python_script_path}'")
        sys.exit(1)


if __name__ == "__main__":
    main()