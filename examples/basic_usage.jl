#!/usr/bin/env julia

"""
# Basic Usage Example for BarrenPlateausVQE.jl (Enhanced Visualization Edition)

This script demonstrates the complete functionality of the BarrenPlateausVQE.jl package
with enhanced visualization capabilities, including 3D loss landscapes and optimization trajectories.

Run this script with:
```bash
julia examples/basic_usage.jl
```
"""

using Pkg
Pkg.activate(".")

using BarrenPlateausVQE
using Printf
using Statistics
using LinearAlgebra

println("🚀 BarrenPlateausVQE.jl Basic Usage Example (Enhanced Visualization)")
println("=" ^ 70)
println("This example demonstrates:")
println("• Single method VQE analysis")
println("• Multi-method comparison")
println("• Enhanced energy convergence plots")
println("• 3D loss landscape visualization")
println("• Optimization trajectory analysis")
println("=" ^ 70)

# ============================================================================
# Example 1: Single Method Analysis with Visualization (H₂ molecule)
# ============================================================================

println("\n📋 Example 1: Single Method Analysis with Enhanced Visualization")
println("-" ^ 60)

# Create H₂ Hamiltonian
println("Creating H₂ Hamiltonian...")
h2_system = create_molecular_hamiltonian("H2", geometry="equilibrium", basis="sto-3g")
println("✓ H₂ system created: $(h2_system.n_qubits) qubits")
println("  Exact ground state energy: $(round(h2_system.exact_energy, digits=6))")

# Create output directory for plots
output_dir = "./example_plots"
mkpath(output_dir)
println("✓ Output directory created: $output_dir")

# Create and run Standard VQE with more iterations for better trajectory
println("\nRunning Standard VQE with trajectory tracking...")
vqe = StandardVQE(h2_system.n_qubits, 2)  # 2 layers for more parameters
initial_params = random_initial_parameters(vqe.n_parameters)

result = run_vqe(vqe, h2_system.hamiltonian, initial_params, 1000; verbose=false)

println("✓ Standard VQE completed!")
println("  Final energy: $(round(result.final_energy, digits=6))")
println("  Exact energy: $(round(h2_system.exact_energy, digits=6))")
println("  Energy error: $(round(abs(result.final_energy - h2_system.exact_energy), digits=8))")
println("  Converged: $(result.converged)")
println("  Iterations: $(result.num_iterations)")
println("  Parameters: $(vqe.n_parameters)")

# Create single-method visualization
println("\nCreating single-method visualizations...")
try
    # Energy convergence plot
    energies = result.energy_history
    iterations = 1:length(energies)
    
    # Manual plot creation to ensure it works
    using Plots
    
    # Energy convergence
    p1 = plot(iterations, energies, 
             title="H₂ Standard VQE Energy Convergence",
             xlabel="Iterations", 
             ylabel="Energy",
             linewidth=2,
             color=:blue,
             label="Standard VQE",
             size=(800, 500))
    
    # Add exact energy line
    hline!(p1, [h2_system.exact_energy], 
           linestyle=:dash, 
           color=:red, 
           linewidth=2, 
           label="Exact Ground State")
    
    savefig(p1, joinpath(output_dir, "h2_energy_convergence.png"))
    println("  ✓ Energy convergence plot saved")
    
    # Log-scale energy error plot
    energy_errors = abs.(energies .- h2_system.exact_energy)
    energy_errors = max.(energy_errors, 1e-12)  # Avoid log(0)
    
    p2 = plot(iterations, log10.(energy_errors),
             title="H₂ Standard VQE Energy Error (Log Scale)",
             xlabel="Iterations",
             ylabel="log₁₀(|Energy - Exact|)",
             linewidth=2,
             color=:purple,
             label="Energy Error",
             size=(800, 500))
    
    savefig(p2, joinpath(output_dir, "h2_energy_error_log.png"))
    println("  ✓ Log-scale energy error plot saved")
    
    # 3D Loss landscape visualization
    if vqe.n_parameters >= 2
        println("  🧮 Computing 3D loss landscape...")
        
        # Create cost function
        function h2_cost_function(params::Vector{Float64})
            return energy_evaluation(h2_system.hamiltonian, vqe.ansatz, params, h2_system.n_qubits)
        end
        
        # Use enhanced visualization functions
        landscape_3d = plot_loss_landscape_3d(
            h2_cost_function, result.final_parameters;
            param_indices=(1,2),
            param_range=0.4,
            resolution=25,
            show_trajectory=true,
            parameter_history=result.parameter_history,
            save_path=joinpath(output_dir, "h2_loss_landscape_3d.png")
        )
        println("  ✓ 3D loss landscape saved")
        
        # Contour landscape
        landscape_contour = plot_loss_landscape_contour(
            h2_cost_function, result.final_parameters;
            param_indices=(1,2),
            param_range=0.4,
            resolution=40,
            show_trajectory=true,
            parameter_history=result.parameter_history,
            save_path=joinpath(output_dir, "h2_loss_landscape_contour.png")
        )
        println("  ✓ Contour landscape saved")
        
        # Trajectory analysis
        traj_i, traj_j = compute_optimization_trajectory_2d(result.parameter_history; param_indices=(1,2))
        trajectory_length = sqrt(sum(diff(traj_i).^2 + diff(traj_j).^2))
        println("  📊 Optimization trajectory length: $(round(trajectory_length, digits=4))")
        println("  📊 Parameter path: $(length(traj_i)) points")
    else
        println("  ⚠️  Insufficient parameters for landscape visualization")
    end
    
catch e
    println("  ❌ Visualization failed: $e")
    println("     This might be due to missing plotting packages")
end

# ============================================================================
# Example 2: Multi-Method Comparison with Enhanced Analysis
# ============================================================================

println("\n📋 Example 2: Multi-Method Comparison with Enhanced Visualization")
println("-" ^ 60)

# Create analyzer for LiH (more complex system)
println("Setting up LiH analysis...")
analyzer = MolecularVQEAnalyzer("LiH", geometry="equilibrium", n_layers=2)
println("✓ LiH analyzer created: $(analyzer.n_qubits) qubits, $(analyzer.n_layers) layers")
println("  Exact ground state energy: $(round(analyzer.exact_energy, digits=6))")

# Run comprehensive analysis with more methods
methods_to_test = ["standard", "local_global", "sea"]
println("\nRunning comprehensive VQE analysis...")
println("  Methods: $(join(methods_to_test, ", "))")
println("  Iterations per method: 400")

results = run_complete_analysis(analyzer, 
                               num_iters=400, 
                               methods=methods_to_test, 
                               verbose=false)

println("✓ Multi-method comparison completed!")

# Enhanced results analysis
println("\n📊 Enhanced Results Analysis:")
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
            final_energy_avg = mean(energies[end-9:end])
            improvement = initial_energy - final_energy_avg
            stability = std(energies[end-19:end])
        else
            improvement = energies[1] - energies[end]
            stability = std(energies[max(1, end-9):end])
        end
        
        push!(performance_summary, (
            method = method_name,
            final_energy = final_energy,
            energy_error = energy_error,
            improvement = improvement,
            stability = stability,
            grad_var = grad_var,
            exec_time = exec_time
        ))
        
        # Print detailed results
        println("🔍 $method_name:")
        println("  Final energy: $(round(final_energy, digits=6))")
        println("  Energy error: $(round(energy_error, digits=8))")
        println("  Energy improvement: $(round(improvement, digits=6))")
        println("  Final stability (σ): $(round(stability, digits=8))")
        println("  Gradient variance: $(round(grad_var, digits=8))")
        println("  Execution time: $(round(exec_time, digits=2))s")
    end
end

# ============================================================================
# Example 3: Enhanced Visualization Suite
# ============================================================================

println("\n📋 Example 3: Enhanced Visualization Suite")
println("-" ^ 60)

println("Creating comprehensive visualization suite...")
try
    # 1. Enhanced energy convergence comparison
    conv_plot = plot_energy_convergence(analyzer, 
                                       save_path=joinpath(output_dir, "lih_energy_convergence.png"),
                                       show_exact=true)
    println("✓ Multi-method energy convergence plot saved")
    
    # 2. Log-scale convergence for detailed analysis
    conv_log_plot = plot_energy_convergence(analyzer, 
                                           log_scale=true,
                                           save_path=joinpath(output_dir, "lih_energy_convergence_log.png"))
    println("✓ Log-scale convergence plot saved")
    
    # 3. Enhanced quick analysis plot
    quick_plot = quick_analysis_plot(analyzer)
    try
        savefig(quick_plot, joinpath(output_dir, "lih_quick_analysis.png"))
        println("✓ Quick analysis plot saved")
    catch
        println("✓ Quick analysis plot generated (display only)")
    end
    
    # 4. Performance comparison table
    df = create_performance_table(analyzer, 
                                 save_csv=joinpath(output_dir, "lih_performance_table.csv"))
    if df !== nothing
        println("✓ Performance table saved")
    end
    
    # 5. Method-specific loss landscapes
    landscape_count = 0
    for (method_name, data) in analyzer.results
        if get(data["method_result"], "fallback", false) || landscape_count >= 2
            continue  # Limit to 2 landscapes to save time
        end
        
        try
            vqe_result = data["method_result"]["vqe_result"]
            ansatz = data["method_result"]["ansatz_info"]["ansatz"]
            final_params = vqe_result.final_parameters
            param_history = vqe_result.parameter_history
            
            if length(final_params) >= 2
                println("  🧮 Creating loss landscape for $method_name...")
                
                # Create method-specific cost function
                function method_cost_function(params::Vector{Float64})
                    return energy_evaluation(analyzer.hamiltonian, ansatz, params, analyzer.n_qubits)
                end
                
                # Clean method name for filename
                clean_name = replace(method_name, " " => "_", "/" => "_")
                
                # 3D landscape
                plot_loss_landscape_3d(
                    method_cost_function, final_params;
                    param_indices=(1,2),
                    param_range=0.3,
                    resolution=20,
                    show_trajectory=true,
                    parameter_history=param_history,
                    save_path=joinpath(output_dir, "lih_landscape_3d_$(clean_name).png")
                )
                
                # Contour landscape
                plot_loss_landscape_contour(
                    method_cost_function, final_params;
                    param_indices=(1,2),
                    param_range=0.3,
                    resolution=30,
                    show_trajectory=true,
                    parameter_history=param_history,
                    save_path=joinpath(output_dir, "lih_landscape_contour_$(clean_name).png")
                )
                
                println("    ✓ Landscapes saved for $method_name")
                landscape_count += 1
            end
        catch e
            println("    ❌ Landscape creation failed for $method_name: $e")
        end
    end
    
catch e
    println("❌ Enhanced visualization failed: $e")
    println("   Basic analysis still available in results")
end

# ============================================================================
# Example 4: Loss Landscape Analysis
# ============================================================================

println("\n📋 Example 4: Detailed Loss Landscape Analysis")
println("-" ^ 60)

# Analyze the loss landscape characteristics
println("Analyzing loss landscape characteristics...")
try
    # Use the best performing method for detailed analysis
    best_method = nothing
    best_error = Inf
    
    for (method_name, data) in analyzer.results
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
        println("🏆 Best performing method: $method_name (error: $(round(best_error, digits=8)))")
        
        vqe_result = data["method_result"]["vqe_result"]
        ansatz = data["method_result"]["ansatz_info"]["ansatz"]
        final_params = vqe_result.final_parameters
        
        # Create cost function
        function best_cost_function(params::Vector{Float64})
            return energy_evaluation(analyzer.hamiltonian, ansatz, params, analyzer.n_qubits)
        end
        
        # Compute landscape statistics
        println("  🧮 Computing landscape statistics...")
        param_i_range, param_j_range, landscape = compute_loss_landscape_2d(
            best_cost_function, final_params;
            param_indices=(1,2),
            param_range=0.4,
            resolution=30
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
            println("    Landscape ruggedness: $(round(energy_std/energy_range, digits=4))")
        end
        
        # Trajectory analysis
        if !isempty(vqe_result.parameter_history)
            traj_i, traj_j = compute_optimization_trajectory_2d(
                vqe_result.parameter_history; param_indices=(1,2)
            )
            
            if length(traj_i) > 1
                # Calculate trajectory metrics
                trajectory_distances = sqrt.(diff(traj_i).^2 + diff(traj_j).^2)
                total_distance = sum(trajectory_distances)
                avg_step_size = mean(trajectory_distances)
                
                # Parameter space exploration
                param_range_i = maximum(traj_i) - minimum(traj_i)
                param_range_j = maximum(traj_j) - minimum(traj_j)
                
                println("  📊 Trajectory Analysis:")
                println("    Total path length: $(round(total_distance, digits=4))")
                println("    Average step size: $(round(avg_step_size, digits=6))")
                println("    Parameter exploration (param 1): $(round(param_range_i, digits=4))")
                println("    Parameter exploration (param 2): $(round(param_range_j, digits=4))")
                
                # Convergence analysis
                energies = vqe_result.energy_history
                if length(energies) > 10
                    early_energies = energies[1:div(length(energies), 3)]
                    late_energies = energies[div(2*length(energies), 3):end]
                    
                    early_mean = mean(early_energies)
                    late_mean = mean(late_energies)
                    improvement_rate = (early_mean - late_mean) / length(energies)
                    
                    println("    Improvement rate: $(round(improvement_rate, digits=8)) per iteration")
                end
            end
        end
    end
    
catch e
    println("❌ Landscape analysis failed: $e")
end

# ============================================================================
# Example 5: Barren Plateau Detection
# ============================================================================

println("\n📋 Example 5: Barren Plateau Detection Analysis")
println("-" ^ 60)

println("Analyzing barren plateau phenomena...")
try
    barren_analysis = []
    
    for (method_name, data) in analyzer.results
        if !get(data["method_result"], "fallback", false)
            bp_diag = data["bp_diagnostics"]
            vqe_result = data["method_result"]["vqe_result"]
            
            grad_var = bp_diag.gradient_variance
            grad_norm = bp_diag.gradient_norm_mean
            
            # Barren plateau indicators
            # Low gradient variance and norm suggest barren plateaus
            barren_indicator = grad_var < 1e-6 && grad_norm < 1e-3
            
            # Energy landscape flatness
            energies = vqe_result.energy_history
            if length(energies) > 50
                # Check for plateaus in energy (periods of little change)
                energy_changes = abs.(diff(energies))
                plateau_threshold = 1e-8
                plateau_steps = sum(energy_changes .< plateau_threshold)
                plateau_fraction = plateau_steps / length(energy_changes)
            else
                plateau_fraction = 0.0
            end
            
            push!(barren_analysis, (
                method = method_name,
                grad_var = grad_var,
                grad_norm = grad_norm,
                barren_indicator = barren_indicator,
                plateau_fraction = plateau_fraction
            ))
            
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
        println("\n⚠️  High barren plateau risk detected in: $(join(high_risk_methods, ", "))")
        println("   Consider using barren plateau mitigation strategies")
    else
        println("\n✓ Low barren plateau risk across all methods")
    end
    
catch e
    println("❌ Barren plateau analysis failed: $e")
end

# ============================================================================
# Summary and File Overview
# ============================================================================

println("\n🎉 Enhanced Basic Usage Example Completed!")
println("=" ^ 70)

# List generated files
println("\n📁 Generated Visualization Files:")
if isdir(output_dir)
    files = readdir(output_dir)
    for file in sort(files)
        filepath = joinpath(output_dir, file)
        filesize = round(stat(filepath).size / 1024, digits=1)
        println("  📊 $file ($(filesize) KB)")
    end
    println("\n  📂 Location: $output_dir")
else
    println("  ⚠️  No output directory created")
end

# Performance summary
println("\n📊 Performance Summary:")
if !isempty(performance_summary)
    # Find best performer in each category
    best_energy = minimum([x.energy_error for x in performance_summary])
    best_speed = minimum([x.exec_time for x in performance_summary])
    best_stability = minimum([x.stability for x in performance_summary])
    
    for perf in performance_summary
        indicators = String[]
        if perf.energy_error == best_energy
            push!(indicators, "🎯 Most Accurate")
        end
        if perf.exec_time == best_speed
            push!(indicators, "⚡ Fastest")
        end
        if perf.stability == best_stability
            push!(indicators, "📈 Most Stable")
        end
        
        indicator_str = isempty(indicators) ? "" : " " * join(indicators, " ")
        println("  $(perf.method): Error $(round(perf.energy_error, digits=6))$indicator_str")
    end
end