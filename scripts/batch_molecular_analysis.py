#!/usr/bin/env python3
"""
Batch Molecular Analysis Helper Script
=====================================

Helper script to run batch analyses for multiple molecules and configurations.
Provides predefined configurations and automated batch processing.

Usage:
    python3 scripts/batch_molecular_analysis.py --preset small_molecules
    python3 scripts/batch_molecular_analysis.py --preset layer_scaling
    python3 scripts/batch_molecular_analysis.py --custom H2,LiH --iterations 500
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_available_presets():
    """Get available analysis presets."""
    return {
        "quick_test": {
            "description": "Quick test with H2 molecule",
            "molecules": [
                {"molecule": "H2", "geometry": "equilibrium", "basis": "sto-3g"}
            ],
            "iterations": 200,
            "layer_scaling": False,
        },
        "small_molecules": {
            "description": "Analysis of small molecules",
            "molecules": [
                {"molecule": "H2", "geometry": "equilibrium", "basis": "sto-3g"},
                {"molecule": "H2", "geometry": "stretched", "basis": "sto-3g"},
                {"molecule": "LiH", "geometry": "equilibrium", "basis": "sto-3g"},
                {"molecule": "BeH2", "geometry": "equilibrium", "basis": "sto-3g"},
            ],
            "iterations": 1000,
            "layer_scaling": False,
        },
        "layer_scaling": {
            "description": "Layer scaling analysis for H2",
            "molecules": [
                {"molecule": "H2", "geometry": "equilibrium", "basis": "sto-3g"}
            ],
            "iterations": 500,
            "layer_scaling": True,
            "max_layers": 6,
        },
        "basis_comparison": {
            "description": "Basis set comparison for H2",
            "molecules": [
                {"molecule": "H2", "geometry": "equilibrium", "basis": "sto-3g"},
                {"molecule": "H2", "geometry": "equilibrium", "basis": "6-31g"},
                {"molecule": "H2", "geometry": "equilibrium", "basis": "cc-pvdz"},
            ],
            "iterations": 800,
            "layer_scaling": False,
        },
        "geometry_effects": {
            "description": "Geometry effects for multiple molecules",
            "molecules": [
                {"molecule": "H2", "geometry": "equilibrium", "basis": "sto-3g"},
                {"molecule": "H2", "geometry": "stretched", "basis": "sto-3g"},
                {"molecule": "LiH", "geometry": "equilibrium", "basis": "sto-3g"},
                {"molecule": "LiH", "geometry": "stretched", "basis": "sto-3g"},
            ],
            "iterations": 800,
            "layer_scaling": False,
        },
        "large_molecules": {
            "description": "Large molecules with active space",
            "molecules": [
                {
                    "molecule": "H2O",
                    "geometry": "equilibrium",
                    "basis": "sto-3g",
                    "active_space": [8, 6],
                },
                {
                    "molecule": "BeH2",
                    "geometry": "equilibrium",
                    "basis": "sto-3g",
                    "active_space": [4, 6],
                },
            ],
            "iterations": 600,
            "layer_scaling": False,
        },
        "comprehensive": {
            "description": "Comprehensive analysis suite",
            "molecules": [
                {"molecule": "H2", "geometry": "equilibrium", "basis": "sto-3g"},
                {"molecule": "H2", "geometry": "stretched", "basis": "sto-3g"},
                {"molecule": "LiH", "geometry": "equilibrium", "basis": "sto-3g"},
                {"molecule": "BeH2", "geometry": "equilibrium", "basis": "sto-3g"},
                {
                    "molecule": "H2O",
                    "geometry": "equilibrium",
                    "basis": "sto-3g",
                    "active_space": [8, 6],
                },
            ],
            "iterations": 1000,
            "layer_scaling": True,
            "max_layers": 4,
        },
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch molecular barren plateau analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    presets = get_available_presets()

    parser.add_argument(
        "--preset", choices=list(presets.keys()), help="Use predefined analysis preset"
    )

    parser.add_argument(
        "--custom", help="Custom molecule list (comma-separated): H2,LiH,BeH2"
    )

    parser.add_argument(
        "--iterations", type=int, help="Override iterations for analysis"
    )

    parser.add_argument(
        "--layer-scaling", action="store_true", help="Enable layer scaling analysis"
    )

    parser.add_argument(
        "--max-layers", type=int, default=4, help="Maximum layers for scaling"
    )

    parser.add_argument(
        "--basis", default="sto-3g", help="Basis set for custom molecules"
    )

    parser.add_argument(
        "--geometry", default="equilibrium", help="Geometry for custom molecules"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run analyses in parallel (experimental)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
    )

    parser.add_argument(
        "--list-presets", action="store_true", help="List available presets and exit"
    )

    return parser.parse_args()


def list_presets():
    """List available analysis presets."""
    presets = get_available_presets()

    print("Available Analysis Presets:")
    print("=" * 50)

    for preset_name, config in presets.items():
        print(f"\n{preset_name}:")
        print(f"  Description: {config['description']}")
        print(f"  Molecules: {len(config['molecules'])}")
        for mol in config["molecules"]:
            mol_str = f"{mol['molecule']} ({mol['geometry']}, {mol['basis']}"
            if "active_space" in mol:
                mol_str += f", AS: {mol['active_space']}"
            mol_str += ")"
            print(f"    - {mol_str}")
        print(f"  Iterations: {config['iterations']}")
        print(f"  Layer scaling: {config.get('layer_scaling', False)}")
        if config.get("layer_scaling"):
            print(f"  Max layers: {config.get('max_layers', 4)}")


def create_analysis_jobs(args):
    """Create list of analysis jobs based on arguments."""
    jobs = []

    if args.preset:
        # Use preset configuration
        presets = get_available_presets()
        config = presets[args.preset]

        for mol_config in config["molecules"]:
            job = {
                "molecule": mol_config["molecule"],
                "geometry": mol_config["geometry"],
                "basis": mol_config["basis"],
                "iterations": args.iterations or config["iterations"],
                "layer_scaling": args.layer_scaling
                or config.get("layer_scaling", False),
                "max_layers": args.max_layers or config.get("max_layers", 4),
            }

            # Add active space if specified
            if "active_space" in mol_config:
                job["active_space"] = mol_config["active_space"]

            jobs.append(job)

    elif args.custom:
        # Create custom jobs
        molecules = [mol.strip() for mol in args.custom.split(",")]

        for molecule in molecules:
            job = {
                "molecule": molecule,
                "geometry": args.geometry,
                "basis": args.basis,
                "iterations": args.iterations or 1000,
                "layer_scaling": args.layer_scaling,
                "max_layers": args.max_layers,
            }
            jobs.append(job)

    else:
        print("❌ Either --preset or --custom must be specified")
        return []

    return jobs


def build_command(job, job_id):
    """Build command line for analysis job."""
    script_path = Path(__file__).parent / "molecular_bp_analysis.py"

    cmd = [
        "python",
        str(script_path),
        "--molecule",
        job["molecule"],
        "--geometry",
        job["geometry"],
        "--basis",
        job["basis"],
        "--iterations",
        str(job["iterations"]),
        "--output-suffix",
        f"batch_{job_id}",
    ]

    if job.get("active_space"):
        cmd.extend(["--active-space"] + [str(x) for x in job["active_space"]])

    if job["layer_scaling"]:
        cmd.append("--layer-scaling")
        cmd.extend(["--max-layers", str(job["max_layers"])])

    return cmd


def run_analysis_job(job, job_id, dry_run=False):
    """Run single analysis job."""
    cmd = build_command(job, job_id)

    print(f"\n🚀 Job {job_id}: {job['molecule']} ({job['geometry']}, {job['basis']})")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("   [DRY RUN - not executed]")
        return True

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )  # 1 hour timeout

        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}...")
            return False

    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout (>1 hour)")
        return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def run_batch_analysis(jobs, parallel=False, dry_run=False):
    """Run batch analysis jobs."""
    print(f"\n🔄 Running batch analysis...")
    print(f"Total jobs: {len(jobs)}")
    print(f"Parallel: {parallel}")
    print(f"Dry run: {dry_run}")

    if parallel and not dry_run:
        print("⚠️  Parallel execution is experimental and may cause resource conflicts")

    results = []
    start_time = datetime.now()

    if parallel and not dry_run:
        # Parallel execution using subprocess
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = {}
            for i, job in enumerate(jobs):
                future = executor.submit(run_analysis_job, job, i + 1, dry_run)
                futures[future] = (i + 1, job)

            for future in concurrent.futures.as_completed(futures):
                job_id, job = futures[future]
                try:
                    success = future.result()
                    results.append((job_id, job, success))
                except Exception as e:
                    print(f"Job {job_id} generated exception: {e}")
                    results.append((job_id, job, False))

    else:
        # Sequential execution
        for i, job in enumerate(jobs):
            success = run_analysis_job(job, i + 1, dry_run)
            results.append((i + 1, job, success))

    end_time = datetime.now()
    duration = end_time - start_time

    # Print summary
    print(f"\n📊 BATCH ANALYSIS SUMMARY")
    print("=" * 50)

    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful

    print(f"Total jobs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Duration: {duration}")

    if not dry_run:
        print(f"\nJob Details:")
        for job_id, job, success in results:
            status = "✅" if success else "❌"
            print(
                f"  {status} Job {job_id}: {job['molecule']} ({job['geometry']}, {job['basis']})"
            )

    # Save batch summary
    if not dry_run:
        root_dir = Path(__file__).parent.parent
        batch_summary = {
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "total_jobs": len(results),
            "successful_jobs": successful,
            "failed_jobs": failed,
            "jobs": [
                {
                    "job_id": job_id,
                    "molecule": job["molecule"],
                    "geometry": job["geometry"],
                    "basis": job["basis"],
                    "success": success,
                }
                for job_id, job, success in results
            ],
        }

        summary_file = (
            root_dir
            / "data"
            / f"batch_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, "w") as f:
            json.dump(batch_summary, f, indent=2)
        print(f"\n💾 Batch summary saved: {summary_file}")

    return results


def main():
    """Main batch analysis function."""
    print("🧪 Batch Molecular Barren Plateau Analysis")
    print("=" * 60)

    args = parse_arguments()

    # List presets if requested
    if args.list_presets:
        list_presets()
        return 0

    # Validate arguments
    if not args.preset and not args.custom:
        print("❌ Either --preset or --custom must be specified")
        print("Use --list-presets to see available options")
        return 1

    # Create analysis jobs
    jobs = create_analysis_jobs(args)

    if not jobs:
        print("❌ No jobs created")
        return 1

    print(f"\n📋 Created {len(jobs)} analysis jobs:")
    for i, job in enumerate(jobs):
        mol_str = f"{job['molecule']} ({job['geometry']}, {job['basis']}"
        if "active_space" in job:
            mol_str += f", AS: {job['active_space']}"
        mol_str += ")"
        scaling_str = f", {job['max_layers']} layers" if job["layer_scaling"] else ""
        print(f"  {i+1}. {mol_str} - {job['iterations']} iter{scaling_str}")

    # Run batch analysis
    results = run_batch_analysis(jobs, args.parallel, args.dry_run)

    successful = sum(1 for _, _, success in results if success)

    if not args.dry_run:
        if successful == len(jobs):
            print(f"\n🎉 All {len(jobs)} jobs completed successfully!")
        elif successful > 0:
            print(f"\n⚠️  {successful}/{len(jobs)} jobs completed successfully")
        else:
            print(f"\n❌ All jobs failed")

        print(f"\nCheck the following directories for results:")
        print(f"  data/ - Analysis data and summaries")
        print(f"  plots/ - Generated visualizations")

    return 0 if successful > 0 or args.dry_run else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
