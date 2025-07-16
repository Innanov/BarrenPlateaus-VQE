#!/usr/bin/env julia

"""
# Basic Usage Example for BarrenPlateausVQE.jl

This script demonstrates the basic functionality of the BarrenPlateausVQE.jl package,
showing how to analyze barren plateau phenomena in molecular VQE systems.

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

println("🚀 BarrenPlateausVQE.jl Basic Usage Example")
println("=" ^ 60)

# ============================================================================
# Example 1: Single Method Analysis (H₂ molecule)
# ============================================================================

println("\n📋 Example 1: Single Method Analysis")
println("-" ^ 40)

# Create H₂ Hamiltonian
println("Creating H₂ Hamiltonian...")
h2_system = create_molecular_hamiltonian("H2", geometry="equilibrium", basis="sto-3g")
println("✓ H₂ system created: $(h2_system.n_qubits) qubits")

# Create and run Standard VQE
println("\nRunning Standard VQE...")
vqe = StandardVQE(h2_system.n_qubits, 2)  # 2 layers
initial_params = random_initial_parameters(vqe.n_parameters)

result = run_vqe(vqe, h2_system.hamiltonian, initial_params, 200; verbose=false)

println("✓ Standard VQE completed!")
println("  Final energy: $(round(result.final_energy, digits=6))")
println("  Exact energy: $(round(h2_system.exact_energy, digits=6))")
println("  Energy error: $(round(abs(result.final_energy - h2_system.exact_energy), digits=8))")
println("  Converged: $(result.converged)")
println("  Iterations: $(result.num_iterations)")

# ============================================================================
# Example 2: Method Comparison (LiH molecule)
# ============================================================================

println("\n📋 Example 2: Method Comparison")
println("-" ^ 40)

# Create analyzer for LiH
println("Setting up LiH analysis...")
analyzer = MolecularVQEAnalyzer("LiH", geometry="equilibrium", n_layers=2)
println("✓ LiH analyzer created: $(analyzer.n_qubits) qubits, $(analyzer.n_layers) layers")

# Run subset of methods for quick demonstration
methods_to_test = ["standard", "local_global", "sea"]
println("\nRunning VQE methods: $(join(methods_to_test, ", "))")

results = run_complete_analysis(analyzer, 
                               num_iters=300, 
                               methods=methods_to_test, 
                               verbose=false)

println("✓ Method comparison completed!")

# Display results with proper error handling
println("\n📊 Results Summary:")
for (method_name, data) in results
    if !get(data["method_result"], "fallback", false)
        final_energy = data["method_result"]["vqe_result"].final_energy
        energy_error = data["performance_metrics"]["final_energy_error"]
        grad_var = data["bp_diagnostics"].gradient_variance
        exec_time = data["execution_time"]
        
        # Use @sprintf correctly
        result_line = @sprintf("  %-15s | Energy: %8.6f | Error: %.2e | Grad Var: %.2e | Time: %.1fs", 
                              method_name, final_energy, energy_error, grad_var, exec_time)
        println(result_line)
    end
end

# ============================================================================
# Example 3: Visualization
# ============================================================================

println("\n📋 Example 3: Visualization")
println("-" ^ 40)

# Generate quick analysis plot
println("Generating visualization...")
try
    quick_plot = quick_analysis_plot(analyzer)
    println("✓ Quick analysis plot generated")
    
    # Create performance table
    df = create_performance_table(analyzer)
    if df !== nothing
        println("✓ Performance table created")
    else
        println("⚠️  Performance table creation skipped")
    end
    
    # Save results
    output_dir = "./example_output"
    save_results(analyzer, output_dir)
    save_analysis_summary(analyzer, output_dir)
    println("✓ Results saved to: $output_dir")
    
catch e
    println("⚠️  Visualization issue: $e")
    println("   This is normal in headless environments or without proper plotting backends")
end

# ============================================================================
# Example 4: Individual VQE Methods
# ============================================================================

println("\n📋 Example 4: Individual VQE Method Examples")
println("-" ^ 40)

# Demonstrate each method individually
h2_hamiltonian = h2_system.hamiltonian
h2_local = global2local(h2_hamiltonian, h2_system.n_qubits)

println("Testing individual VQE methods on H₂...")

# 1. Standard VQE
println("\n1. Standard VQE:")
try
    vqe_std = StandardVQE(2, 1)
    params_std = random_initial_parameters(vqe_std.n_parameters, seed=100)
    result_std = run_vqe(vqe_std, h2_hamiltonian, params_std, 100, verbose=false)
    println("   Final energy: $(round(result_std.final_energy, digits=6))")
catch e
    println("   ❌ Failed: $e")
end

# 2. Local-Global VQE
println("\n2. Local-Global VQE:")
try
    vqe_lg = LocalGlobalVQE(2, 1)
    params_lg = random_initial_parameters(vqe_lg.n_parameters, seed=200)
    results_lg = run_vqe(vqe_lg, h2_local, h2_hamiltonian, params_lg, 100, 30, verbose=false)
    final_energy_lg = results_lg["out"]["final_energy"]
    println("   Final energy: $(round(final_energy_lg, digits=6))")
catch e
    println("   ❌ Failed: $e")
end

# 3. Adiabatic VQE
println("\n3. Adiabatic VQE:")
try
    vqe_ad = AdiabaticVQE(2, 1)
    params_ad = random_initial_parameters(vqe_ad.n_parameters, seed=300)
    result_ad = run_vqe(vqe_ad, h2_local, h2_hamiltonian, params_ad, 100, verbose=false)
    println("   Final energy: $(round(result_ad.final_energy, digits=6))")
catch e
    println("   ❌ Failed: $e")
end

# 4. SEA VQE
println("\n4. SEA VQE:")
try
    vqe_sea = SEAVQE(2, depth=[1, 1, 1])
    params_sea = random_initial_parameters(vqe_sea.n_parameters, seed=400)
    result_sea = run_vqe(vqe_sea, h2_hamiltonian, params_sea, 100, verbose=false)
    println("   Final energy: $(round(result_sea.final_energy, digits=6))")
catch e
    println("   ❌ Failed: $e")
end

# 5. Pretrained VQE
println("\n5. Pretrained VQE:")
try
    vqe_pre = PretrainedVQE(2)
    results_pre = run_vqe(vqe_pre, h2_hamiltonian, 80, 20, verbose=false)
    final_energy_pre = results_pre["full"]["final_energy"]
    println("   Final energy: $(round(final_energy_pre, digits=6))")
catch e
    println("   ❌ Failed: $e")
end

# ============================================================================
# Example 5: Advanced Features
# ============================================================================

println("\n📋 Example 5: Advanced Features")
println("-" ^ 40)

# Custom Hamiltonian
println("Creating custom Hamiltonian...")
try
    custom_terms = [("ZZ", 0.5), ("XX", -0.3), ("ZI", 0.1), ("IZ", 0.1)]
    custom_H = create_pauli_hamiltonian(2, custom_terms)
    exact_energy_custom = classical_solver(custom_H).eigenvalue
    println("✓ Custom Hamiltonian created, exact energy: $(round(exact_energy_custom, digits=6))")

    # Test with custom Hamiltonian
    vqe_custom = StandardVQE(2, 1)
    params_custom = random_initial_parameters(vqe_custom.n_parameters, seed=500)
    result_custom = run_vqe(vqe_custom, custom_H, params_custom, 150, verbose=false)
    custom_error = abs(result_custom.final_energy - exact_energy_custom)
    println("✓ VQE on custom Hamiltonian: error = $(round(custom_error, digits=8))")

    # Gradient analysis
    println("\nGradient analysis example...")
    function cost_func(params)
        return energy_evaluation(custom_H, vqe_custom.ansatz, params, 2)
    end

    try
        gradient = gradient_finite_diff(cost_func, result_custom.final_parameters)
        grad_variance = var(gradient)
        grad_norm = norm(gradient)
        println("✓ Gradient analysis:")
        println("   Gradient variance: $(round(grad_variance, digits=8))")
        println("   Gradient norm: $(round(grad_norm, digits=8))")
    catch grad_e
        println("⚠️  Gradient analysis failed: $grad_e")
        println("   This might be due to numerical issues")
    end

catch custom_e
    println("❌ Custom Hamiltonian example failed: $custom_e")
end

# ============================================================================
# Example 6: Simple Convergence Analysis
# ============================================================================

println("\n📋 Example 6: Simple Convergence Analysis")
println("-" ^ 40)

try
    # Analyze convergence of different methods
    println("Comparing convergence characteristics...")
    
    convergence_data = []
    
    for (method_name, data) in analyzer.results
        if !get(data["method_result"], "fallback", false)
            energies = data["method_result"]["vqe_result"].energy_history
            final_energy = data["method_result"]["vqe_result"].final_energy
            
            # Simple convergence metrics
            if length(energies) > 10
                early_energy = mean(energies[1:min(10, end)])
                late_energy = mean(energies[max(1, end-9):end])
                improvement = early_energy - late_energy
                stability = std(energies[max(1, end-9):end])
                
                push!(convergence_data, (
                    method = method_name,
                    improvement = improvement,
                    stability = stability,
                    final_energy = final_energy
                ))
            end
        end
    end
    
    if !isempty(convergence_data)
        println("✓ Convergence analysis:")
        for data in convergence_data
            println(@sprintf("   %-15s | Improvement: %8.6f | Stability: %.2e | Final: %8.6f",
                           data.method, data.improvement, data.stability, data.final_energy))
        end
    else
        println("⚠️  No convergence data available")
    end

catch conv_e
    println("⚠️  Convergence analysis failed: $conv_e")
end

# ============================================================================
# Summary
# ============================================================================

println("\n🎉 Basic Usage Example Completed!")
println("=" ^ 60)
println("Key takeaways:")
println("✓ Single method VQE optimization")
println("✓ Multi-method comparison analysis") 
println("✓ Visualization and result export")
println("✓ Individual method testing")
println("✓ Custom Hamiltonian usage")
println("✓ Gradient analysis capabilities")
println("✓ Convergence analysis")
println("")
println("Next steps:")
println("• Try examples/molecular_systems.jl for more molecular examples")
println("• Try examples/scaling_study.jl for performance analysis")
println("• Check the documentation for advanced features")
println("• Run your own molecular systems with different parameters")
println("")
println("🚀 Ready for your quantum algorithm research!")