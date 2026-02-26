#!/usr/bin/env julia

"""
# Molecular VQE Analysis Script

Comprehensive analysis tool for Variational Quantum Eigensolvers (VQE) with multiple methods,
enhanced visualization, and barren plateau detection.

## Usage

```bash
julia scripts/molecular_vqe_analysis.jl --molecule H2 --layers 2 --iterations 1000 --methods standard,sea,adiabatic
```

## Arguments

- `--molecule, -m`: Molecule to analyze (default: H2)
- `--geometry, -g`: Molecular geometry (default: equilibrium)
- `--basis, -b`: Basis set (default: sto-3g)
- `--layers, -l`: Number of ansatz layers (default: 2)
- `--iterations, -i`: Number of optimization iterations (default: 1000)
- `--methods`: Comma-separated list of VQE methods (default: all)
- `--output-dir, -o`: Base output directory (default: plots)
- `--landscape-resolution`: Resolution for landscape plots (default: 30)
- `--landscape-range`: Parameter range for landscape analysis (default: 0.4)
- `--verbose, -v`: Verbose output (flag)
- `--help, -h`: Show this help message

## Available Methods

- standard: Standard VQE
- local_global: Local-Global VQE
- sea: Simultaneous Evolution Ansatz
- adiabatic: Adiabatic VQE
- pretrained: Pre-trained VQE

## Output Structure

```
plots/
└── YYYY-MM-DD_HH-MM-SS_molecule_layers/
    ├── energy_convergence_all_methods.pdf
    ├── loss_landscape_3d_standard.pdf
    ├── loss_landscape_3d_sea.pdf
    ├── loss_landscape_3d_adiabatic.pdf
    ├── loss_landscape_3d_local_global.pdf
    ├── loss_landscape_3d_pretrained.pdf
    ├── loss_landscape_contour_standard.pdf
    ├── loss_landscape_contour_sea.pdf
    ├── loss_landscape_contour_adiabatic.pdf
    ├── loss_landscape_contour_local_global.pdf
    ├── loss_landscape_contour_pretrained.pdf
    ├── performance_comparison_table.csv
    ├── analysis_summary.json
    └── run_parameters.json
```
"""

using Pkg
Pkg.activate(".")

using ArgParse
using BarrenPlateausVQE
using Printf
using Statistics
using LinearAlgebra
using Dates
using JSON3

function parse_commandline()
    s = ArgParseSettings(;
        prog="molecular_vqe_analysis.jl",
        description="Comprehensive Molecular VQE Analysis Tool",
        version="1.0.0",
        add_version=true,
    )

    @add_arg_table! s begin
        "--molecule", "-m"
        help = "Molecule to analyze (e.g., H2, LiH, BeH2)"
        arg_type = String
        default = "H2"

        "--geometry", "-g"
        help = "Molecular geometry configuration"
        arg_type = String
        default = "equilibrium"

        "--basis", "-b"
        help = "Basis set for molecular calculations"
        arg_type = String
        default = "STO-3G"

        "--layers", "-l"
        help = "Number of ansatz layers"
        arg_type = Int
        default = 1

        "--iterations", "-i"
        help = "Number of optimization iterations per method"
        arg_type = Int
        default = 100

        "--methods"
        help = "Comma-separated list of VQE methods to test (standard,local_global,sea,adiabatic,pretrained) or 'all'"
        arg_type = String
        default = "all"

        "--output-dir", "-o"
        help = "Base output directory for results"
        arg_type = String
        default = "results"

        "--landscape-resolution"
        help = "Resolution for loss landscape plots"
        arg_type = Int
        default = 100

        "--landscape-range"
        help = "Parameter range for loss landscape analysis"
        arg_type = Float64
        default = 0.4

        "--verbose", "-v"
        help = "Enable verbose output"
        action = :store_true
    end

    return parse_args(s)
end

function setup_output_directory(base_dir::String, molecule::String, layers::Int)
    """Create structured output directory: <base>/<molecule>/vqe_analysis/<timestamp>/"""
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    run_id = "$(timestamp)_$(layers)layers"
    output_dir = joinpath(base_dir, molecule, "vqe_analysis", run_id)

    mkpath(output_dir)
    return output_dir, run_id
end

function save_run_parameters(output_dir::String, args::Dict, additional_info::Dict=Dict())
    """Save run parameters and configuration to JSON file."""
    run_params = merge(args, additional_info)
    run_params["timestamp"] = string(now())
    run_params["julia_version"] = string(VERSION)

    open(joinpath(output_dir, "run_parameters.json"), "w") do f
        write(f, JSON3.write(run_params; allow_inf=true))
    end
end

function parse_methods(methods_str::String)
    """Parse methods string into array, handling 'all' keyword."""
    all_methods = ["standard", "local_global", "sea", "adiabatic", "pretrained"]

    if lowercase(methods_str) == "all"
        return all_methods
    else
        methods = [strip(m) for m in split(methods_str, ",")]
        # Validate methods
        invalid_methods = setdiff(methods, all_methods)
        if !isempty(invalid_methods)
            error(
                "Invalid methods: $(join(invalid_methods, ", ")). Available: $(join(all_methods, ", "))",
            )
        end
        return methods
    end
end

function generate_analysis_summary(
    analyzer, results, performance_summary, output_dir::String, molecule_name::String
)
    """Generate comprehensive analysis summary and save to JSON."""

    summary = Dict(
        "system_info" => Dict(
            "molecule" => molecule_name,
            "n_qubits" => analyzer.n_qubits,
            "n_layers" => analyzer.n_layers,
            "exact_energy" => analyzer.exact_energy,
        ),
        "methods_analyzed" => length(results),
        "results" => Dict(),
        "best_performers" => Dict(),
    )

    if !isempty(performance_summary)
        # Find best performers
        best_accuracy = argmin([x.energy_error for x in performance_summary])
        best_speed = argmin([x.exec_time for x in performance_summary])
        best_stability = argmin([x.stability for x in performance_summary])

        summary["best_performers"] = Dict(
            "most_accurate" => Dict(
                "method" => performance_summary[best_accuracy].method,
                "energy_error" => performance_summary[best_accuracy].energy_error,
            ),
            "fastest" => Dict(
                "method" => performance_summary[best_speed].method,
                "execution_time" => performance_summary[best_speed].exec_time,
            ),
            "most_stable" => Dict(
                "method" => performance_summary[best_stability].method,
                "stability" => performance_summary[best_stability].stability,
            ),
        )

        # Detailed results for each method
        for perf in performance_summary
            summary["results"][perf.method] = Dict(
                "final_energy" => perf.final_energy,
                "energy_error" => perf.energy_error,
                "improvement" => perf.improvement,
                "stability" => perf.stability,
                "gradient_variance" => perf.grad_var,
                "execution_time" => perf.exec_time,
            )
        end
    end

    # Save summary
    open(joinpath(output_dir, "analysis_summary.json"), "w") do f
        write(f, JSON3.write(summary; allow_inf=true))
    end

    return summary
end

function print_header(args::Dict)
    """Print formatted header with run information."""
    println("🚀 Molecular VQE Analysis Script")
    println("=" ^ 70)
    println("📋 Run Configuration:")
    println("  Molecule: $(args["molecule"])")
    println("  Geometry: $(args["geometry"])")
    println("  Basis set: $(args["basis"])")
    println("  Ansatz layers: $(args["layers"])")
    println("  Iterations: $(args["iterations"])")
    println("  Methods: $(args["methods"])")
    println("  Output directory: $(args["output-dir"])")
    println("  Timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("=" ^ 70)
end

function main()
    # Parse command line arguments
    args = parse_commandline()

    # Print header
    print_header(args)

    # Setup output directory
    output_dir, run_id = setup_output_directory(
        args["output-dir"], args["molecule"], args["layers"]
    )
    println("📁 Created output directory: $output_dir")

    # Parse methods
    methods_to_test = parse_methods(args["methods"])
    println("🔬 Methods to analyze: $(join(methods_to_test, ", "))")

    try
        # ============================================================================
        # System Setup
        # ============================================================================

        println("\n📋 Setting up molecular system...")

        # Create molecular Hamiltonian
        molecular_system = create_molecular_hamiltonian(
            args["molecule"]; geometry=args["geometry"], basis=args["basis"]
        )

        println("✓ $(args["molecule"]) system created:")
        println("  Qubits: $(molecular_system.n_qubits)")
        println(
            "  Exact ground state energy: $(round(molecular_system.exact_energy, digits=8))"
        )

        # Create analyzer
        analyzer = MolecularVQEAnalyzer(
            args["molecule"]; geometry=args["geometry"], n_layers=args["layers"]
        )

        println("🧪 Initialized MolecularVQEAnalyzer:")
        println("   Molecule: $(args["molecule"])")
        println("   Geometry: $(args["geometry"])")
        println("   Qubits: $(analyzer.n_qubits)")
        println("   Layers: $(analyzer.n_layers)")
        println("   Exact energy: $(analyzer.exact_energy)")

        # Save run parameters with full molecule info
        system_info = Dict(
            "n_qubits"      => molecular_system.n_qubits,
            "n_electrons"   => molecular_system.n_electrons,
            "exact_energy"  => molecular_system.exact_energy,
            "basis"         => molecular_system.basis_set,
            "geometry"      => molecular_system.geometry_type,
            "nuclear_repulsion_energy" => molecular_system.nuclear_repulsion_energy,
            "analysis_type" => "vqe_analysis",
        )
        save_run_parameters(output_dir, args, system_info)

        # ============================================================================
        # VQE Analysis
        # ============================================================================

        println("\n🔬 Running VQE analysis...")
        println("  Methods: $(join(methods_to_test, ", "))")
        println("  Iterations per method: $(args["iterations"])")

        # Run comprehensive analysis
        results = run_complete_analysis(
            analyzer;
            num_iters=args["iterations"],
            methods=methods_to_test,
            verbose=args["verbose"],
        )

        println("✓ VQE analysis completed!")

        # ============================================================================
        # Results Analysis
        # ============================================================================

        println("\n📊 Analyzing results...")
        performance_summary = []

        for (method_name, data) in results
            if !get(data["method_result"], "fallback", false)
                vqe_result = data["method_result"]["vqe_result"]
                perf_metrics = data["performance_metrics"]
                bp_diag = data["bp_diagnostics"]

                final_energy = vqe_result.final_energy
                energy_error = perf_metrics["final_energy_error"]
                grad_var = bp_diag.gradient_variance
                exec_time = data["execution_time"]

                # Calculate additional metrics
                energies = vqe_result.energy_history
                if length(energies) > 20
                    initial_energy = mean(energies[1:10])
                    final_energy_avg = mean(energies[(end - 9):end])
                    improvement = initial_energy - final_energy_avg
                    stability = std(energies[(end - 19):end])
                else
                    improvement = energies[1] - energies[end]
                    stability = std(energies[max(1, end - 9):end])
                end

                push!(
                    performance_summary,
                    (
                        method=method_name,
                        final_energy=final_energy,
                        energy_error=energy_error,
                        improvement=improvement,
                        stability=stability,
                        grad_var=grad_var,
                        exec_time=exec_time,
                    ),
                )

                if args["verbose"]
                    println("🔍 $method_name:")
                    println("  Final energy: $(round(final_energy, digits=8))")
                    println("  Energy error: $(round(energy_error, digits=10))")
                    println("  Execution time: $(round(exec_time, digits=2))s")
                end
            end
        end

        # ============================================================================
        # Visualization Generation
        # ============================================================================

        println("\n🎨 Generating visualizations...")

        visualization_files = []

        try
            # Energy convergence plot (all methods)
            conv_file = joinpath(output_dir, "energy_convergence_all_methods.pdf")
            plot_all_methods_energy_convergence(analyzer; save_path=conv_file)
            push!(visualization_files, "energy_convergence_all_methods.pdf")
            println("✓ Energy convergence plot saved")

            # 3D loss landscapes for all methods
            println("Creating 3D loss landscapes...")
            try
                # Try the original approach first
                landscape_3d_files = plot_all_methods_loss_landscape_3d(
                    analyzer; save_path=joinpath(output_dir, "loss_landscape_3d_dummy.pdf")
                )
                println("✓ 3D loss landscapes function called")

                # Check what files were actually created
                landscape_files_found = false
                for method in methods_to_test
                    expected_file = "loss_landscape_3d_$(method).pdf"
                    if isfile(joinpath(output_dir, expected_file))
                        push!(visualization_files, expected_file)
                        landscape_files_found = true
                        if args["verbose"]
                            println("  ✓ Found 3D landscape: $expected_file")
                        end
                    end
                end

                if !landscape_files_found
                    println(
                        "  ⚠️  No individual 3D landscape files found, checking for combined file...",
                    )
                    # List all PDF files in output directory
                    if args["verbose"]
                        println("  📁 Files in output directory:")
                        for file in readdir(output_dir)
                            if endswith(file, ".pdf")
                                println("    - $file")
                            end
                        end
                    end
                    # Check if a combined file was created
                    if isfile(joinpath(output_dir, "loss_landscape_3d_dummy.pdf"))
                        push!(visualization_files, "loss_landscape_3d_dummy.pdf")
                        println("  ✓ Found combined 3D landscape file")
                    end
                end

            catch e
                println("⚠️  3D landscape creation failed: $e")
            end

            # Contour landscapes for all methods
            println("Creating contour loss landscapes...")
            try
                landscape_contour_files = plot_all_methods_loss_landscape_contour(
                    analyzer;
                    save_path=joinpath(output_dir, "loss_landscape_contour_dummy.pdf"),
                )
                println("✓ Contour landscapes function called")

                # Check what files were actually created
                contour_files_found = false
                for method in methods_to_test
                    expected_file = "loss_landscape_contour_$(method).pdf"
                    if isfile(joinpath(output_dir, expected_file))
                        push!(visualization_files, expected_file)
                        contour_files_found = true
                        if args["verbose"]
                            println("  ✓ Found contour landscape: $expected_file")
                        end
                    end
                end

                if !contour_files_found
                    println(
                        "  ⚠️  No individual contour landscape files found, checking for combined file...",
                    )
                    # Check if a combined file was created
                    if isfile(joinpath(output_dir, "loss_landscape_contour_dummy.pdf"))
                        push!(visualization_files, "loss_landscape_contour_dummy.pdf")
                        println("  ✓ Found combined contour landscape file")
                    end
                end

            catch e
                println("⚠️  Contour landscape creation failed: $e")
            end

            # Performance comparison table
            table_file = joinpath(output_dir, "performance_comparison_table.csv")
            df = create_performance_table_all_methods(analyzer; save_csv=table_file)
            if df !== nothing
                push!(visualization_files, "performance_comparison_table.csv")
                println("✓ Performance comparison table saved")
            end

        catch e
            println("⚠️  Some visualizations failed: $e")
            println("   Results and analysis still available")
        end

        # ============================================================================
        # Detailed Loss Landscape Analysis
        # ============================================================================

        println("\n📋 Detailed Loss Landscape Analysis")
        println("-" ^ 60)

        # Analyze the loss landscape characteristics
        println("Analyzing loss landscape characteristics...")
        try
            # Use the best performing method for detailed analysis
            best_method = nothing
            best_error = Inf

            for (method_name, data) in results
                if !get(data["method_result"], "fallback", false)
                    error = data["performance_metrics"]["final_energy_error"]
                    if error < best_error
                        best_error = error
                        best_method = (method_name, data)
                    end
                end
            end

            if best_method !== nothing
                method_name, data = best_method
                println(
                    "🏆 Best performing method: $method_name (error: $(round(best_error, digits=8)))",
                )

                vqe_result = data["method_result"]["vqe_result"]
                ansatz = data["method_result"]["ansatz_info"]["ansatz"]
                final_params = vqe_result.final_parameters

                # Create cost function
                function best_cost_function(params::Vector{Float64})
                    return energy_evaluation(
                        analyzer.hamiltonian, ansatz, params, analyzer.n_qubits
                    )
                end

                # Compute landscape statistics
                println("  🧮 Computing landscape statistics...")
                param_i_range, param_j_range, landscape = compute_loss_landscape_2d(
                    best_cost_function,
                    final_params;
                    param_indices=(1, 2),
                    param_range=args["landscape-range"],
                    resolution=args["landscape-resolution"],
                )

                # Landscape analysis
                valid_energies = landscape[.!isnan.(landscape)]
                if !isempty(valid_energies)
                    min_energy = minimum(valid_energies)
                    max_energy = maximum(valid_energies)
                    energy_range = max_energy - min_energy
                    energy_std = std(valid_energies)

                    println("  📊 Landscape Statistics:")
                    println("    Energy range: $(round(energy_range, digits=6))")
                    println("    Energy std: $(round(energy_std, digits=6))")
                    println("    Min energy: $(round(min_energy, digits=6))")
                    println("    Max energy: $(round(max_energy, digits=6))")
                    println(
                        "    Landscape ruggedness: $(round(energy_std/energy_range, digits=4))",
                    )
                end

                # Trajectory analysis
                if !isempty(vqe_result.parameter_history)
                    traj_i, traj_j = compute_optimization_trajectory_2d(
                        vqe_result.parameter_history; param_indices=(1, 2)
                    )

                    if length(traj_i) > 1
                        # Calculate trajectory metrics
                        trajectory_distances = sqrt.(diff(traj_i) .^ 2 + diff(traj_j) .^ 2)
                        total_distance = sum(trajectory_distances)
                        avg_step_size = mean(trajectory_distances)

                        # Parameter space exploration
                        param_range_i = maximum(traj_i) - minimum(traj_i)
                        param_range_j = maximum(traj_j) - minimum(traj_j)

                        println("  📊 Trajectory Analysis:")
                        println("    Total path length: $(round(total_distance, digits=4))")
                        println("    Average step size: $(round(avg_step_size, digits=6))")
                        println(
                            "    Parameter exploration (param 1): $(round(param_range_i, digits=4))",
                        )
                        println(
                            "    Parameter exploration (param 2): $(round(param_range_j, digits=4))",
                        )

                        # Convergence analysis
                        energies = vqe_result.energy_history
                        if length(energies) > 10
                            early_energies = energies[1:div(length(energies), 3)]
                            late_energies = energies[div(2 * length(energies), 3):end]

                            early_mean = mean(early_energies)
                            late_mean = mean(late_energies)
                            improvement_rate = (early_mean - late_mean) / length(energies)

                            println(
                                "    Improvement rate: $(round(improvement_rate, digits=8)) per iteration",
                            )
                        end
                    end
                end
            end

        catch e
            println("❌ Landscape analysis failed: $e")
        end

        # ============================================================================
        # Barren Plateau Detection
        # ============================================================================

        println("\n📋 Barren Plateau Detection Analysis")
        println("-" ^ 60)

        println("Analyzing barren plateau phenomena...")
        try
            barren_analysis = []

            for (method_name, data) in results
                if !get(data["method_result"], "fallback", false)
                    bp_diag = data["bp_diagnostics"]
                    vqe_result = data["method_result"]["vqe_result"]

                    grad_var = bp_diag.gradient_variance
                    grad_norm = bp_diag.gradient_norm_mean

                    # Barren plateau indicators
                    barren_indicator = grad_var < 1e-6 && grad_norm < 1e-3

                    # Energy landscape flatness
                    energies = vqe_result.energy_history
                    if length(energies) > 50
                        energy_changes = abs.(diff(energies))
                        plateau_threshold = 1e-8
                        plateau_steps = sum(energy_changes .< plateau_threshold)
                        plateau_fraction = plateau_steps / length(energy_changes)
                    else
                        plateau_fraction = 0.0
                    end

                    push!(
                        barren_analysis,
                        (
                            method=method_name,
                            grad_var=grad_var,
                            grad_norm=grad_norm,
                            barren_indicator=barren_indicator,
                            plateau_fraction=plateau_fraction,
                        ),
                    )

                    println("🔍 $method_name Barren Plateau Analysis:")
                    println("  Gradient variance: $(round(grad_var, digits=10))")
                    println("  Gradient norm: $(round(grad_norm, digits=8))")
                    println("  Plateau fraction: $(round(plateau_fraction*100, digits=2))%")
                    println("  Barren plateau risk: $(barren_indicator ? "HIGH" : "LOW")")
                end
            end

            # Overall assessment
            high_risk_methods = [x.method for x in barren_analysis if x.barren_indicator]
            if !isempty(high_risk_methods)
                println(
                    "\n⚠️  High barren plateau risk detected in: $(join(high_risk_methods, ", "))",
                )
                println("   Consider using barren plateau mitigation strategies")
            else
                println("\n✓ Low barren plateau risk across all methods")
            end

        catch e
            println("❌ Barren plateau analysis failed: $e")
        end

        # ============================================================================
        # Generate Summary
        # ============================================================================

        summary = generate_analysis_summary(
            analyzer, results, performance_summary, output_dir, args["molecule"]
        )

        # ============================================================================
        # Final Report
        # ============================================================================

        println("\n🎉 Analysis Complete!")
        println("=" ^ 70)

        println("\n📊 Performance Summary:")
        if !isempty(performance_summary)
            # Find best performers
            best_energy_idx = argmin([x.energy_error for x in performance_summary])
            best_speed_idx = argmin([x.exec_time for x in performance_summary])
            best_stability_idx = argmin([x.stability for x in performance_summary])

            for (i, perf) in enumerate(performance_summary)
                indicators = String[]
                if i == best_energy_idx
                    push!(indicators, "🎯 Most Accurate")
                end
                if i == best_speed_idx
                    push!(indicators, "⚡ Fastest")
                end
                if i == best_stability_idx
                    push!(indicators, "📈 Most Stable")
                end

                indicator_str = isempty(indicators) ? "" : " " * join(indicators, " ")
                println(
                    "  $(perf.method): Error $(round(perf.energy_error, digits=8))$indicator_str",
                )
            end
        end

        println("\n📁 Output Files:")
        println("  📂 Directory: $output_dir")
        for file in ["run_parameters.json", "analysis_summary.json"]
            if isfile(joinpath(output_dir, file))
                println("  📄 $file")
            end
        end
        for file in visualization_files
            println("  📊 $file")
        end

        println("\n✅ Analysis completed successfully!")
        println("   Run ID: $run_id")

    catch e
        println("\n❌ Analysis failed with error: $e")
        println("   Check parameters and try again")
        rethrow(e)
    end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
