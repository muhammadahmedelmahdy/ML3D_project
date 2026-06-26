"""
Full pipeline: Qwen3-8B layout proposal → TRELLIS RePaint generation → evaluation.

Usage:
  python run_llm_trellis.py \\
      --category chair \\
      --partnet_root /data/PartNet-Mobility \\
      --trellis_model microsoft/TRELLIS-text-large \\
      --ss_encoder_path microsoft/TRELLIS-text-large/ckpts/ss_enc_conv3d_16l8_fp16 \\
      --output_dir outputs/ \\
      [--n_examples 5] \\
      [--steps 250] \\
      [--num_resample_steps 10] \\
      [--seed 42] \\
      [--formats mesh gaussian] \\
      [--evaluate] \\
      [--render_video]

Pipeline steps:
  1. Load PartNet-Mobility context examples for the requested category.
  2. Prompt Qwen3-8B (thinking mode) → JSON layout (parts + description).
  3. Convert bounding boxes → 64³ voxel mask + box-skeleton mesh.
  4. Run TRELLIS text-to-3D with RePaint conditioning on the box skeleton.
  5. Save outputs (mesh OBJ / Gaussian PLY / GLB / video).
  6. [optional] Evaluate layout fidelity (IoU) and shape quality (Chamfer).
"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3 → TRELLIS RePaint pipeline")
    p.add_argument("--category", required=True,
                   help="PartNet-Mobility category (e.g. 'chair')")
    p.add_argument("--partnet_root", required=True,
                   help="Root directory of the PartNet-Mobility dataset")
    p.add_argument("--trellis_model", default="microsoft/TRELLIS-text-large",
                   help="TRELLIS model path or HF hub ID")
    p.add_argument("--ss_encoder_path",
                   default="microsoft/TRELLIS-text-large/ckpts/ss_enc_conv3d_16l8_fp16",
                   help="Sparse-structure encoder checkpoint path")
    p.add_argument("--qwen_model", default="Qwen/Qwen3-8B",
                   help="Qwen3 model name or local path")
    p.add_argument("--output_dir", default="outputs",
                   help="Directory to save generated assets")
    p.add_argument("--n_examples", type=int, default=5,
                   help="Number of PartNet-Mobility in-context examples")
    p.add_argument("--steps", type=int, default=250,
                   help="TRELLIS denoising steps")
    p.add_argument("--num_resample_steps", type=int, default=10,
                   help="RePaint resampling iterations per timestep (U)")
    p.add_argument("--cfg_strength", type=float, default=7.5,
                   help="Classifier-free guidance strength")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formats", nargs="+",
                   default=["mesh", "gaussian"],
                   choices=["mesh", "gaussian", "radiance_field"],
                   help="Output formats to decode from TRELLIS")
    p.add_argument("--save_layout", action="store_true",
                   help="Save the Qwen-proposed layout JSON alongside outputs")
    p.add_argument("--layout_json", default=None,
                   help="Skip Qwen and load a pre-existing layout JSON instead")
    p.add_argument("--evaluate", action="store_true",
                   help="Run layout IoU + Chamfer evaluation after generation")
    p.add_argument("--n_refs", type=int, default=5,
                   help="Number of PartNet-Mobility GT objects for Chamfer (--evaluate)")
    p.add_argument("--render_video", action="store_true",
                   help="Render a 360° turntable video of the generated asset")
    p.add_argument("--video_frames", type=int, default=300,
                   help="Number of frames for the turntable video")
    p.add_argument("--export_glb", action="store_true",
                   help="Export textured GLB (requires both 'mesh' and 'gaussian' formats)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build context from PartNet-Mobility ──────────────────────────
    if args.layout_json is None:
        print(f"[1/4] Loading PartNet-Mobility context for '{args.category}' …")
        from llm.context_builder import PartNetMobilityContextBuilder
        builder = PartNetMobilityContextBuilder(args.partnet_root)
        examples_str = builder.format_examples_for_prompt(
            args.category, n=args.n_examples, seed=args.seed
        )
        print(f"      Loaded {args.n_examples} examples.")
    else:
        examples_str = None

    # ── Step 2: Qwen3-8B layout proposal ─────────────────────────────────────
    if args.layout_json is not None:
        print(f"[2/4] Loading pre-existing layout from {args.layout_json} …")
        with open(args.layout_json) as f:
            layout = json.load(f)
    else:
        print(f"[2/4] Prompting Qwen3-8B for a '{args.category}' layout …")
        from llm.qwen_proposer import QwenLayoutProposer
        proposer = QwenLayoutProposer(model_name=args.qwen_model)
        layout = proposer.propose(
            category=args.category,
            examples_str=examples_str,
            n_examples=args.n_examples,
        )
        print(f"      Description: {layout['description']}")
        print(f"      Parts ({len(layout['parts'])}):")
        for part in layout["parts"]:
            print(f"        {part['label']:20s}  "
                  f"min={part['bbox_min']}  max={part['bbox_max']}")

    if args.save_layout or args.layout_json is None:
        layout_path = out_dir / "layout.json"
        with open(layout_path, "w") as f:
            json.dump(layout, f, indent=2)
        print(f"      Layout saved to {layout_path}")

    # ── Step 3: Boxes → voxel mask + mesh ────────────────────────────────────
    print("[3/4] Converting bounding boxes to voxel mask …")
    from llm.box_to_voxel import layout_to_trellis_inputs
    prompt, box_mesh, structure_mask_3d = layout_to_trellis_inputs(
        layout, device="cuda"
    )
    print(f"      Prompt: '{prompt}'")
    occupied = int(structure_mask_3d.sum().item())
    print(f"      Occupied voxels in mask: {occupied} / {64**3}")

    # ── Step 4: TRELLIS RePaint ───────────────────────────────────────────────
    print(f"[4/4] Running TRELLIS RePaint (steps={args.steps}, "
          f"U={args.num_resample_steps}) …")
    from trellis.pipelines import TrellisTextTo3DPipeline

    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.trellis_model)
    pipeline.cuda()
    pipeline.load_sparse_structure_encoder(args.ss_encoder_path)

    outputs = pipeline.run_inpaint(
        prompt=prompt,
        mesh=box_mesh,
        structure_mask_3d=structure_mask_3d,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.steps,
            "cfg_strength": args.cfg_strength,
            "num_resample_steps": args.num_resample_steps,
        },
        slat_sampler_params={
            "steps": args.steps,
            "cfg_strength": args.cfg_strength,
        },
        formats=args.formats,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    mesh_result = None
    gaussian_result = None

    if "mesh" in outputs:
        from evaluate import save_mesh_obj
        mesh_result = outputs["mesh"][0]
        mesh_path = str(out_dir / "output.obj")
        save_mesh_obj(mesh_result, mesh_path)
        print(f"      Mesh saved to {mesh_path}")

    if "gaussian" in outputs:
        gaussian_result = outputs["gaussian"][0]
        gs_path = str(out_dir / "output.ply")
        gaussian_result.save_ply(gs_path)
        print(f"      Gaussians saved to {gs_path}")

    if "radiance_field" in outputs:
        rf_path = str(out_dir / "output_rf.npz")
        rf = outputs["radiance_field"][0]
        np.savez(rf_path, **{k: v.cpu().numpy() for k, v in rf.__dict__.items()
                              if isinstance(v, torch.Tensor)})
        print(f"      Radiance field saved to {rf_path}")

    # ── Optional: textured GLB export ────────────────────────────────────────
    if args.export_glb:
        if mesh_result is not None and gaussian_result is not None:
            from trellis.utils.postprocessing_utils import to_glb
            print("      Exporting textured GLB …")
            glb_mesh = to_glb(gaussian_result, mesh_result, verbose=True)
            glb_path = str(out_dir / "output.glb")
            glb_mesh.export(glb_path)
            print(f"      GLB saved to {glb_path}")
        else:
            print("      Warning: --export_glb requires both 'mesh' and 'gaussian' "
                  "in --formats. Skipping.")

    # ── Optional: 360° turntable video ───────────────────────────────────────
    if args.render_video:
        import imageio
        from trellis.utils.render_utils import render_video

        # Prefer Gaussian for rendering (faster, higher quality)
        render_target = gaussian_result if gaussian_result is not None else mesh_result
        if render_target is not None:
            print(f"      Rendering {args.video_frames}-frame turntable video …")
            frames = render_video(render_target, num_frames=args.video_frames)
            video_path = str(out_dir / "output.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"      Video saved to {video_path}")
        else:
            print("      Warning: no decodable output for video rendering.")

    # ── Optional: evaluation ──────────────────────────────────────────────────
    if args.evaluate:
        if mesh_result is None and "mesh" not in args.formats:
            print("\nWarning: --evaluate requires 'mesh' in --formats. Skipping.")
        else:
            print("\n[Eval] Running evaluation …")
            from evaluate import evaluate
            eval_target = mesh_result if mesh_result is not None else (out_dir / "output.obj")
            metrics = evaluate(
                generated_mesh=eval_target,
                proposed_mask_3d=structure_mask_3d,
                partnet_root=args.partnet_root,
                category=args.category,
                n_refs=args.n_refs,
                n_samples=10_000,
                seed=args.seed,
            )
            metrics_path = out_dir / "metrics.json"
            metrics_out = {k: v for k, v in metrics.items() if k != "chamfer_all"}
            metrics_out["chamfer_all"] = metrics["chamfer_all"]
            with open(metrics_path, "w") as f:
                json.dump(metrics_out, f, indent=2)
            print(f"\n[Eval] Metrics saved to {metrics_path}")
            print(f"[Eval] layout_iou   = {metrics['layout_iou']:.4f}")
            if metrics["chamfer_mean"] is not None:
                print(f"[Eval] chamfer_mean = {metrics['chamfer_mean']:.6f}")
                print(f"[Eval] chamfer_min  = {metrics['chamfer_min']:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
