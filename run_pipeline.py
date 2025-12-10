import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

from scripts.run_pipeline_a import PARAMS as PARAMS_A, pipeline_a
from scripts.run_pipeline_c import PARAMS as PARAMS_C, pipeline_c
from src.evaluation import collect_metrics_row, finalize_metrics, init_run
from src.io_utils import list_images, read_bgr, save_bgr, save_gray
from src.preprocessing import compute_and_save_background
from src.roi import create_and_save_masks


def save_visuals(viz_dir: Path, img_stem: str, results: Dict) -> None:
    """Persist standard visualization layers if present."""
    if 'edges' in results:
        save_gray(viz_dir / f"{img_stem}_edges.png", results['edges'])
    if 'mask' in results:
        save_gray(viz_dir / f"{img_stem}_mask.png", results['mask'])
    if 'overlay' in results:
        save_bgr(viz_dir / f"{img_stem}_overlay.png", results['overlay'])
    if 'enhanced' in results:
        save_bgr(viz_dir / f"{img_stem}_enhanced.png", results['enhanced'])


def generate_c_background(images: Iterable[Path], out_dir: Path) -> Tuple[Path, object]:
    """Compute and save the median background for pipeline C."""
    bg_img = compute_and_save_background(images, out_dir, subdir="c_background")
    return out_dir / "viz" / "c_background" / "median_background.png", bg_img


def generate_c_roi(first_img, params: dict, out_dir: Path):
    """Create ROI, water, and sand masks for pipeline C."""
    return create_and_save_masks(first_img, params, out_dir, subdir="c_roi")


def run(
    pipeline_name: str,
    raw_dir: Path,
    out_dir: Path,
    ann_dir: Path,
    params: Dict,
    process_fn: Callable,
    preproc: Callable = None
):
    """Run a pipeline end-to-end: preprocessing, processing, saving, and metrics."""
    images = list_images(raw_dir)
    if not images:
        print(f"No images found in {raw_dir}")
        return None

    gt_dict, gt_points, viz_dir, metrics_dir, csv_rows, mse_data, totals, source = init_run(
        ann_dir, out_dir, pipeline_name
    )
    if source:
        print(f"Loaded annotations from: {source.name}\n")

    extra_ctx = preproc(images, params, out_dir) if preproc else {}
    print(f"Pipeline {pipeline_name}: {len(images)} images")

    for img_path in images:
        img_bgr = read_bgr(img_path)
        results = process_fn(img_bgr, params, extra_ctx)
        save_visuals(viz_dir, img_path.stem, results)

        count = len(results['boxes'])
        row, progress, _ = collect_metrics_row(
            img_name=img_path.name,
            pipeline=pipeline_name,
            count=count,
            centers=results['centers'],
            gt_dict=gt_dict,
            gt_points=gt_points,
            totals=totals,
            mse_data=mse_data,
            radius=20.0
        )
        print(f"[{pipeline_name}] {img_path.name} -> {count}{progress}")
        csv_rows.append(row)

    tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
    fields = ['filename', 'pipeline', 'count', 'gt_count', 'residual',
              'tp', 'fp', 'fn', 'precision', 'recall', 'f1']
    result = finalize_metrics(
        pipeline=pipeline_name,
        metrics_dir=metrics_dir,
        csv_rows=csv_rows,
        mse_data=mse_data,
        tp=tp, fp=fp, fn=fn,
        fields=fields
    )

    print(f"\nPerson-level: P={result['precision']:.3f} R={result['recall']:.3f} "
          f"F1={result['f1']:.3f} | MSE={result['mse']:.3f}")
    print(f"Viz: {viz_dir}")
    print(f"Metrics: {result['metrics_file']} | Summary: {result['summary_file']}")
    return result


def process_fn_a(img_bgr, params_local, _ctx):
    return pipeline_a(img_bgr, params_local)


def preproc_c(images, params_local, out_dir):
    print("Computing median background...")
    _, bg_img = generate_c_background(images, out_dir)
    if bg_img is not None:
        print("Background saved.")
    first_img = read_bgr(images[0])
    roi_mask, water_mask, sand_mask = generate_c_roi(first_img, params_local, out_dir)
    return {
        'bg': bg_img,
        'roi': roi_mask,
        'water': water_mask,
        'sand': sand_mask
    }


def process_fn_c(img_bgr, params_local, ctx):
    return pipeline_c(
        img_bgr,
        params_local,
        ctx.get('roi'),
        ctx.get('water'),
        ctx.get('sand'),
        ctx.get('bg')
    )


def run_B():
    script_path = Path(__file__).parent / "scripts" / "run_pipeline_b.py"
    print(f"Launching pipeline B via {script_path} (uses its internal configuration).")
    try:
        subprocess.run(
            ["python", str(script_path)],
            check=True,
            cwd=Path(__file__).parent
        )
    except subprocess.CalledProcessError as e:
        print(f"Pipeline B failed (continuing): {e}")


def main():
    parser = argparse.ArgumentParser(description="Run counting pipelines.")
    parser.add_argument("--pipeline", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/outputs"))
    parser.add_argument("--ann-dir", type=Path, default=Path("data/annotations"))
    args = parser.parse_args()

    if args.pipeline == "A":
        run(
            pipeline_name="A",
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            ann_dir=args.ann_dir,
            params=PARAMS_A,
            process_fn=process_fn_a,
        )
    elif args.pipeline == "B":
        run_B()
    elif args.pipeline == "C":
        run(
            pipeline_name="C",
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            ann_dir=args.ann_dir,
            params=PARAMS_C,
            process_fn=process_fn_c,
            preproc=preproc_c
        )
    else:
        # all: clear outputs and run A, B, then C, continuing on failures
        if args.out_dir.exists():
            shutil.rmtree(args.out_dir)
        print("Running pipelines A, B, C (outputs cleared).")
        run(
            pipeline_name="A",
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            ann_dir=args.ann_dir,
            params=PARAMS_A,
            process_fn=process_fn_a,
        )
        run_B()
        run(
            pipeline_name="C",
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            ann_dir=args.ann_dir,
            params=PARAMS_C,
            process_fn=process_fn_c,
            preproc=preproc_c
        )


if __name__ == "__main__":
    main()
