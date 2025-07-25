import sys
import argparse
import numpy as np
import torch
import imageio
from pathlib import Path
from typing import List, Dict

from sam2.build_sam import build_sam2_video_predictor  # video API :contentReference[oaicite:0]{index=0}
from utils import save_array_to_img, dilate_mask, show_mask, show_points

def load_video_to_array(path: str) -> np.ndarray:
    """
    Load either an mp4 (via imageio) or a folder of image files
    into a numpy array of shape (T, H, W, 3).
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov"}:
        reader = imageio.get_reader(str(p))
        frames = [frame for frame in reader]
        reader.close()
    elif p.is_dir():
        imgs = sorted(p.glob("*.[pj][pn]g"))  # jpg/png
        frames = [imageio.imread(str(f)) for f in imgs]
    else:
        raise ValueError(f"Could not interpret {path} as video or folder of frames")
    return np.stack(frames, axis=0)

def predict_video_masks(
    video: np.ndarray,
    prompts: List[Dict],
    model_cfg: str,
    checkpoint: str,
    dilate_kernel: int = None,
    device: str = "cuda"
):
    # build and move predictor to device
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)  # :contentReference[oaicite:1]{index=1}
    predictor.to(device)

    video_tensor = torch.from_numpy(video).to(device=device)  # (T, H, W, 3)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_tensor)

        # add each prompt to its frame
        for obj_id, prm in enumerate(prompts):
            coords = np.array(prm["point_coords"], dtype=float)
            labels = np.array(prm["point_labels"], dtype=int)
            frame_idx = int(prm["frame_idx"])
            _, object_ids, masks = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=coords,
                labels=labels
            )  # :contentReference[oaicite:2]{index=2}

        # now propagate through all frames
        all_outputs = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            all_outputs.append((out_frame_idx, out_obj_ids, out_mask_logits))
    return all_outputs

def setup_args(parser):
    parser.add_argument("--input_video", required=True,
                        help="Path to video file (mp4/avi) or folder of frames")
    parser.add_argument("--point_frames", type=int, nargs='+', required=True,
                        help="Frame index for each point prompt")
    parser.add_argument("--point_coords", type=float, nargs='+', required=True,
                        help="Flattened list of W H coords: [x0 y0  x1 y1  ...]")
    parser.add_argument("--point_labels", type=int, nargs='+', required=True,
                        help="Labels (1=foreground, 0=background) per point")
    parser.add_argument("--dilate_kernel_size", type=int, default=None,
                        help="Optional dilate radius to pad masks")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write per-frame mask folders")
    parser.add_argument("--model_cfg", required=True,
                        help="Path to SAM2 model config .yaml")
    parser.add_argument("--sam_ckpt", required=True,
                        help="SAM2 checkpoint .pt")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load video
    video = load_video_to_array(args.input_video)

    # sanity‑check prompt lengths
    n_pts = len(args.point_frames)
    coords = np.array(args.point_coords, dtype=float)
    assert coords.size == 2 * n_pts, "point_coords must have 2 values per frame"
    assert len(args.point_labels) == n_pts, "need one label per point"

    # build prompt dicts
    coords = coords.reshape(-1, 2)
    prompts = []
    for i in range(n_pts):
        prompts.append({
            "frame_idx": args.point_frames[i],
            "point_coords": [coords[i].tolist()],
            "point_labels": [int(args.point_labels[i])]
        })

    # run
    outputs = predict_video_masks(
        video, prompts,
        model_cfg=args.model_cfg,
        checkpoint=args.sam_ckpt,
        dilate_kernel=args.dilate_kernel_size,
        device=device
    )

    # save
    out_base = Path(args.output_dir)
    for frame_idx, obj_ids, mask_logits in outputs:
        frame = video[frame_idx]
        H, W = frame.shape[:2]
        frame_dir = out_base / f"frame_{frame_idx:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for obj_id, logits in zip(obj_ids, mask_logits):
            mask = (logits[0].cpu().numpy() > 0).astype(np.uint8) * 255
            if args.dilate_kernel_size:
                mask = dilate_mask(mask, args.dilate_kernel_size)
            mask_path = frame_dir / f"obj{obj_id}_mask.png"
            save_array_to_img(mask, mask_path)

            # optional visualization with points overlaid
            vis = Path(frame_dir / f"obj{obj_id}_vis.png")
            import matplotlib.pyplot as plt
            dpi = plt.rcParams['figure.dpi']
            plt.figure(figsize=(W/dpi, H/dpi))
            plt.imshow(frame)
            plt.axis('off')
            # find prompt for this object (if any)
            for prm in prompts:
                if prm["frame_idx"] == frame_idx:
                    show_points(plt.gca(), prm["point_coords"], prm["point_labels"],
                                size=(W*0.04)**2)
            show_mask(plt.gca(), mask, random_color=False)
            plt.savefig(vis, bbox_inches="tight", pad_inches=0)
            plt.close()
