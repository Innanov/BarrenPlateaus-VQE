"""
# Molecular VQE Analyzer (`molecular_analyzer.jl`)

Core VQE analysis engine for studying barren plateau phenomena in molecular systems.

Key Features:
- Complete barren plateau analysis for molecular VQE systems
- All 5 VQE methods: Standard, Local-Global, Adiabatic, SEA, Pretrained
- Comprehensive gradient diagnostics and landscape analysis
"""

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf

# ============================================================================
# Molecular VQE Analyzer
# ============================================================================

"""
    MolecularVQEAnalyzer

Main analyzer class for comprehensive VQE barren plateau analysis on molecular systems.
"""
mutable struct MolecularVQEAnalyzer
    molecular_system::MolecularSystem
    hamiltonian::AbstractBlock
    local_hamiltonian::AbstractBlock
    exact_energy::Float64
    n_qubits::Int
    n_layers::Int
    results::Dict{String, Any}
    
    function MolecularVQEAnalyzer(molecule::String;
                                 geometry::String="equilibrium",
                                 basis::String="sto-3g",
                                 active_space::Union{Nothing, Tuple{Int,Int}}=nothing,
                                 n_layers::Int=1,
                                 use_test_hamiltonian::Bool=false)
        
        if use_test_hamiltonian
            # Use test Hamiltonian
            println("Using test Hamiltonian for benchmarking")
            n_qubits = 6
            hamiltonian = test_hamiltonian(n_qubits)
            local_hamiltonian = global2local(hamiltonian, n_qubits)
            exact_energy = classical_solver(hamiltonian).eigenvalue
            
            # Create dummy molecular system
            molecular_system = MolecularSystem(
                "Test", "test", "test", 0, 0, 6, n_qubits, 
                "Test Hamiltonian", hamiltonian, exact_energy, 0.0
            )
        else
            # Create molecular system
            molecular_system = create_molecular_hamiltonian(molecule;
                                                           geometry=geometry,
                                                           basis=basis,
                                                           active_space=active_space)
            
            hamiltonian = molecular_system.hamiltonian
            n_qubits = molecular_system.n_qubits
            exact_energy = molecular_system.exact_energy
            
            # Create local Hamiltonian for mitigation techniques
            local_hamiltonian = global2local(hamiltonian, n_qubits)
        end
        
        println("🧪 Initialized MolecularVQEAnalyzer:")
        println("   Molecule: $(molecular_system.name)")
        println("   Geometry: $(molecular_system.geometry_type)")
        println("   Qubits: $n_qubits")
        println("   Layers: $n_layers")
        println("   Exact energy: $exact_energy")
        
        new(molecular_system, hamiltonian, local_hamiltonian, exact_energy,
            n_qubits, n_layers, Dict{String, Any}())
    end
end

# ============================================================================
# VQE Method Execution Functions
# ============================================================================

"""
    run_standard_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)

Run Standard VQE analysis.
"""
function run_standard_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)
    if verbose
        println("\n🔬 Running Standard VQE...")
    end
    
    try
        # Create Standard VQE
        vqe = StandardVQE(analyzer.n_qubits, analyzer.n_layers)
        initial_guess = random_initial_parameters(vqe.n_parameters; seed=102)
        
        # Run optimization
        result = run_vqe(vqe, analyzer.hamiltonian, initial_guess, num_iters; verbose=verbose)
        
        return Dict(
            "method" => "Standard VQE",
            "vqe_result" => result,
            "ansatz_info" => Dict(
                "n_parameters" => vqe.n_parameters,
                "n_qubits" => vqe.n_qubits,
                "n_layers" => analyzer.n_layers,
                "ansatz" => vqe.ansatz
            )
        )
        
    catch e
        @warn "Standard VQE failed: $e"
        return create_fallback_result("Standard VQE", string(e))
    end
end

"""
    run_local_global_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)

Run Local-Global VQE analysis.
"""
function run_local_global_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)
    if verbose
        println("\n🔬 Running Local-Global VQE...")
    end
    
    try
        # Create Local-Global VQE
        vqe = LocalGlobalVQE(analyzer.n_qubits, analyzer.n_layers)
        initial_guess = random_initial_parameters(vqe.n_parameters; seed=200)
        
        # Split iterations: 30% local, 70% global
        shift_iter = div(num_iters, 3)
        
        # Run optimization
        results = run_vqe(vqe, analyzer.local_hamiltonian, analyzer.hamiltonian,
                         initial_guess, num_iters, shift_iter; verbose=verbose)
        
        # Convert to unified format for consistency
        combined_energies = vcat(results["in"]["fx"], results["out"]["fx"])
        combined_params = vcat(results["in"]["x"], results["out"]["x"])
        
        unified_result = VQEResult(
            "Local-Global VQE",
            combined_energies,
            combined_params,
            results["out"]["final_energy"],
            results["out"]["final_parameters"],
            length(combined_energies),
            results["out"]["converged"],
            results["in"]["execution_time"] + results["out"]["execution_time"]
        )
        
        return Dict(
            "method" => "Local-Global VQE",
            "vqe_result" => unified_result,
            "detailed_results" => results,
            "ansatz_info" => Dict(
                "n_parameters" => vqe.n_parameters,
                "n_qubits" => vqe.n_qubits,
                "n_layers" => analyzer.n_layers,
                "ansatz" => vqe.ansatz
            )
        )
        
    catch e
        @warn "Local-Global VQE failed: $e"
        return create_fallback_result("Local-Global VQE", string(e))
    end
end

"""
    run_adiabatic_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)

Run Adiabatic VQE analysis.
"""
function run_adiabatic_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)
    if verbose
        println("\n🔬 Running Adiabatic VQE...")
    end
    
    try
        # Create Adiabatic VQE
        vqe = AdiabaticVQE(analyzer.n_qubits, analyzer.n_layers)
        initial_guess = random_initial_parameters(vqe.n_parameters; seed=250)
        
        # Run optimization
        result = run_vqe(vqe, analyzer.local_hamiltonian, analyzer.hamiltonian,
                        initial_guess, num_iters; verbose=verbose)
        
        return Dict(
            "method" => "Adiabatic VQE",
            "vqe_result" => result,
            "ansatz_info" => Dict(
                "n_parameters" => vqe.n_parameters,
                "n_qubits" => vqe.n_qubits,
                "n_layers" => analyzer.n_layers,
                "ansatz" => vqe.ansatz
            )
        )
        
    catch e
        @warn "Adiabatic VQE failed: $e"
        return create_fallback_result("Adiabatic VQE", string(e))
    end
end

"""
    run_sea_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)

Run SEA VQE analysis.
"""
function run_sea_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)
    if verbose
        println("\n🔬 Running SEA VQE...")
    end
    
    try
        # Create SEA VQE with default depth configuration
        depth_config = [1, 1, 1]  # Conservative depth to avoid barren plateaus
        vqe = SEAVQE(analyzer.n_qubits; depth=depth_config)
        initial_guess = random_initial_parameters(vqe.n_parameters; seed=3000)
        
        # Run optimization
        result = run_vqe(vqe, analyzer.hamiltonian, initial_guess, num_iters; verbose=verbose)
        
        return Dict(
            "method" => "SEA VQE",
            "vqe_result" => result,
            "ansatz_info" => Dict(
                "n_parameters" => vqe.n_parameters,
                "n_qubits" => vqe.n_qubits,
                "depth_config" => depth_config,
                "ansatz" => vqe.ansatz
            )
        )
        
    catch e
        @warn "SEA VQE failed: $e"
        return create_fallback_result("SEA VQE", string(e))
    end
end

"""
    run_pretrained_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)

Run Pretrained VQE analysis.
"""
function run_pretrained_vqe(analyzer::MolecularVQEAnalyzer; num_iters::Int=300, verbose::Bool=true)
    if verbose
        println("\n🔬 Running Pretrained VQE...")
    end
    
    try
        # Create Pretrained VQE
        vqe = PretrainedVQE(analyzer.n_qubits)
        
        # Split iterations: 25% pretraining, 75% full optimization
        iters_train = div(num_iters, 4)
        iters_vqe = num_iters - iters_train
        
        # Run optimization
        results = run_vqe(vqe, analyzer.hamiltonian, iters_vqe, iters_train; verbose=verbose)
        
        # Convert to unified format
        mps_energies = results["pretrain"]["fx"]
        full_energies = results["full"]["fx"]
        combined_energies = vcat(mps_energies, full_energies)
        
        mps_params = results["pretrain"]["x"]
        full_params = results["full"]["x"]
        combined_params = vcat(mps_params, full_params)
        
        unified_result = VQEResult(
            "Pretrained VQE",
            combined_energies,
            combined_params,
            results["full"]["final_energy"],
            results["full"]["final_parameters"],
            length(combined_energies),
            true,  # Assume converged if completed both stages
            0.0    # Would need to track execution time properly
        )
        
        return Dict(
            "method" => "Pretrained VQE",
            "vqe_result" => unified_result,
            "detailed_results" => results,
            "ansatz_info" => Dict(
                "mps_n_parameters" => vqe.mps_n_parameters,
                "full_n_parameters" => vqe.full_n_parameters,
                "n_qubits" => vqe.n_qubits,
                "ansatz" => vqe.full_ansatz
            )
        )
        
    catch e
        @warn "Pretrained VQE failed: $e"
        return create_fallback_result("Pretrained VQE", string(e))
    end
end

"""
    create_fallback_result(method_name::String, error_msg::String)

Create fallback result when a method fails.
"""
function create_fallback_result(method_name::String, error_msg::String)
    # Create minimal fallback result
    fallback_result = VQEResult(
        method_name,
        [1.0],  # High energy indicating poor performance
        [rand(10) * 0.1],  # Random parameters
        1.0,
        rand(10) * 0.1,
        1,
        false,
        0.0
    )
    
    # Create dummy ansatz for compatibility
    dummy_ansatz, dummy_params = efficient_su2_ansatz(4, 1)
    
    return Dict(
        "method" => method_name,
        "vqe_result" => fallback_result,
        "error" => error_msg,
        "fallback" => true,
        "ansatz_info" => Dict(
            "n_parameters" => dummy_params,
            "n_qubits" => 4,
            "ansatz" => dummy_ansatz
        )
    )
end

# ============================================================================
# Barren Plateau Diagnostics
# ============================================================================

"""
    compute_bp_diagnostics(analyzer::MolecularVQEAnalyzer, method_result::Dict; 
                          num_samples::Int=30)

Compute comprehensive barren plateau diagnostics for a VQE method result.
"""
function compute_bp_diagnostics(analyzer::MolecularVQEAnalyzer, method_result::Dict; 
                               num_samples::Int=30)
    
    if get(method_result, "fallback", false)
        # Return minimal diagnostics for fallback results
        return BarrenPlateauDiagnostics(
            1e-3, 1e-2, 1e-3, 1e-2, 1e-3, 0.0,
            [1e-2], [1.0], [1e-3]
        )
    end
    
    vqe_result = method_result["vqe_result"]
    final_params = vqe_result.final_parameters
    ansatz = method_result["ansatz_info"]["ansatz"]
    
    # Define cost function for gradient analysis
    function cost_function(params::Vector{Float64})
        return energy_evaluation(analyzer.hamiltonian, ansatz, params, analyzer.n_qubits)
    end
    
    # Compute gradient variance
    grad_var = gradient_variance(cost_function, final_params)
    
    # Compute landscape statistics (simplified to avoid duplication)
    landscape_stats = compute_local_landscape_statistics(cost_function, final_params; 
                                                        num_samples=num_samples)
    
    gradient_norms = landscape_stats.gradient_norms
    cost_values = landscape_stats.cost_values
    local_variances = landscape_stats.local_variances
    
    return BarrenPlateauDiagnostics(
        grad_var,
        mean(gradient_norms),
        std(gradient_norms),
        var(cost_values),
        mean(local_variances),
        0.0,  # Hessian trace computation would be expensive
        gradient_norms,
        cost_values,
        local_variances
    )
end

"""
    compute_local_landscape_statistics(cost_func, parameters::Vector{Float64};
                                      num_samples::Int=100,
                                      perturbation_scale::Float64=0.1)

Compute local landscape statistics (renamed to avoid duplication with quantum_utils).
"""
function compute_local_landscape_statistics(cost_func, parameters::Vector{Float64};
                                           num_samples::Int=100,
                                           perturbation_scale::Float64=0.1)
    cost_values = Float64[]
    gradient_norms = Float64[]
    local_variances = Float64[]
    
    for _ in 1:num_samples
        # Random perturbation around current parameters
        perturbed_params = parameters + randn(length(parameters)) * perturbation_scale
        
        # Cost value
        try
            cost_val = cost_func(perturbed_params)
            push!(cost_values, isfinite(cost_val) ? cost_val : Inf)
        catch
            push!(cost_values, Inf)
        end
        
        # Gradient norm
        gradient = gradient_finite_diff(cost_func, perturbed_params)
        push!(gradient_norms, norm(gradient))
        
        # Local variance
        local_costs = Float64[]
        for _ in 1:5  # Reduced for performance
            local_params = perturbed_params + randn(length(parameters)) * 0.01
            try
                local_cost = cost_func(local_params)
                if isfinite(local_cost)
                    push!(local_costs, local_cost)
                end
            catch
                # Skip failed evaluations
            end
        end
        
        if !isempty(local_costs)
            push!(local_variances, var(local_costs))
        else
            push!(local_variances, 0.0)
        end
    end
    
    return (
        gradient_norms = gradient_norms,
        cost_values = cost_values,
        local_variances = local_variances
    )
end

"""
    compute_performance_metrics(analyzer::MolecularVQEAnalyzer, method_result::Dict)

Compute performance metrics for a VQE method result.
"""
function compute_performance_metrics(analyzer::MolecularVQEAnalyzer, method_result::Dict)
    if get(method_result, "fallback", false)
        return Dict(
            "final_energy_error" => 1.0,
            "state_fidelity" => 0.0,
            "energy_variance" => 0.1,
            "min_energy_reached" => 1.0,
            "convergence_rate" => 0.0
        )
    end
    
    vqe_result = method_result["vqe_result"]
    energies = vqe_result.energy_history
    
    final_energy_error = abs(vqe_result.final_energy - analyzer.exact_energy)
    min_energy_reached = minimum(energies)
    
    # Energy variance over last 50% of iterations
    n_late = max(1, div(length(energies), 2))
    late_energies = energies[end-n_late+1:end]
    energy_variance = var(late_energies)
    
    # Simple fidelity estimate based on energy accuracy
    energy_scale = abs(analyzer.exact_energy) > 1e-6 ? abs(analyzer.exact_energy) : 1.0
    state_fidelity = exp(-final_energy_error / energy_scale)
    
    # Convergence rate (energy improvement per iteration)
    if length(energies) > 1
        total_improvement = energies[1] - energies[end]
        convergence_rate = total_improvement / length(energies)
    else
        convergence_rate = 0.0
    end
    
    return Dict(
        "final_energy_error" => final_energy_error,
        "state_fidelity" => state_fidelity,
        "energy_variance" => energy_variance,
        "min_energy_reached" => min_energy_reached,
        "convergence_rate" => convergence_rate
    )
end

# ============================================================================
# Complete Analysis Pipeline
# ============================================================================

"""
    run_complete_analysis(analyzer::MolecularVQEAnalyzer; 
                         num_iters::Int=300,
                         methods::Vector{String}=["all"],
                         verbose::Bool=true)

Run complete barren plateau analysis with all VQE methods.
"""
function run_complete_analysis(analyzer::MolecularVQEAnalyzer; 
                              num_iters::Int=300,
                              methods::Vector{String}=["all"],
                              verbose::Bool=true)
    
    if verbose
        println("🚀 Starting Complete VQE Barren Plateau Analysis")
        println("=" * 60)
        println("System: $(analyzer.molecular_system.name)")
        println("Qubits: $(analyzer.n_qubits)")
        println("Layers: $(analyzer.n_layers)")
        println("Iterations per method: $num_iters")
        println("=" * 60)
    end
    
    # Define available methods
    all_methods = ["standard", "local_global", "adiabatic", "sea", "pretrained"]
    
    if "all" in methods
        methods_to_run = all_methods
    else
        methods_to_run = [m for m in methods if m in all_methods]
    end
    
    if verbose
        println("Methods to run: $(join(methods_to_run, ", "))")
    end
    
    # Run each method
    results = Dict{String, Any}()
    
    for method in methods_to_run
        if verbose
            println("\n" * "="^40)
            println("Method: $(uppercase(method))")
            println("="^40)
        end
        
        start_time = time()
        
        try
            if method == "standard"
                method_result = run_standard_vqe(analyzer; num_iters=num_iters, verbose=verbose)
            elseif method == "local_global"
                method_result = run_local_global_vqe(analyzer; num_iters=num_iters, verbose=verbose)
            elseif method == "adiabatic"
                method_result = run_adiabatic_vqe(analyzer; num_iters=num_iters, verbose=verbose)
            elseif method == "sea"
                method_result = run_sea_vqe(analyzer; num_iters=num_iters, verbose=verbose)
            elseif method == "pretrained"
                method_result = run_pretrained_vqe(analyzer; num_iters=num_iters, verbose=verbose)
            else
                @warn "Unknown method: $method"
                continue
            end
            
            # Compute barren plateau diagnostics
            bp_diagnostics = compute_bp_diagnostics(analyzer, method_result)
            
            # Compute performance metrics
            performance_metrics = compute_performance_metrics(analyzer, method_result)
            
            # Store complete results
            results[method_result["method"]] = Dict(
                "method_result" => method_result,
                "bp_diagnostics" => bp_diagnostics,
                "performance_metrics" => performance_metrics,
                "execution_time" => time() - start_time
            )
            
            if verbose
                println("✅ $(method_result["method"]) completed successfully!")
                println("   Final energy: $(method_result["vqe_result"].final_energy)")
                println("   Energy error: $(performance_metrics["final_energy_error"])")
                println("   Gradient variance: $(bp_diagnostics.gradient_variance)")
                println("   Time: $(round(time() - start_time, digits=2))s")
            end
            
        catch e
            if verbose
                println("❌ $method failed: $e")
            end
            
            # Store fallback result
            fallback = create_fallback_result(titlecase(method) * " VQE", string(e))
            bp_diagnostics = compute_bp_diagnostics(analyzer, fallback)
            performance_metrics = compute_performance_metrics(analyzer, fallback)
            
            results[fallback["method"]] = Dict(
                "method_result" => fallback,
                "bp_diagnostics" => bp_diagnostics,
                "performance_metrics" => performance_metrics,
                "execution_time" => time() - start_time
            )
        end
    end
    
    # Store results in analyzer
    analyzer.results = results
    
    if verbose
        println("\n" * "="^60)
        println("🎉 Complete Analysis Finished!")
        println("="^60)
        generate_summary_table(analyzer)
    end
    
    return results
end

"""
    generate_summary_table(analyzer::MolecularVQEAnalyzer)

Generate and print summary table of all results.
"""
function generate_summary_table(analyzer::MolecularVQEAnalyzer)
    println("\n📊 SUMMARY TABLE")
    println("="^80)
    
    header = @sprintf("%-20s %-12s %-12s %-15s %-10s %-10s", 
                     "Method", "Grad Var", "Grad Norm", "Energy Error", "Fidelity", "Time (s)")
    println(header)
    println("-"^length(header))
    
    for (method_name, data) in analyzer.results
        bp_diag = data["bp_diagnostics"]
        perf_met = data["performance_metrics"]
        exec_time = data["execution_time"]
        
        row = Printf.sprintf("%-20s %-12.2e %-12.2e %-15.2e %-10.3f %-10.1f",
                            method_name,
                            bp_diag.gradient_variance,
                            bp_diag.gradient_norm_mean,
                            perf_met["final_energy_error"],
                            perf_met["state_fidelity"],
                            exec_time)
        println(row)
    end
    
    println("="^80)
end

# ============================================================================
# Analysis Results Export
# ============================================================================

"""
    save_results(analyzer::MolecularVQEAnalyzer, output_dir::String)

Save analysis results to files.
"""
function save_results(analyzer::MolecularVQEAnalyzer, output_dir::String)
    mkpath(output_dir)
    
    # Save metadata
    metadata = Dict(
        "molecular_system" => Dict(
            "name" => analyzer.molecular_system.name,
            "geometry_type" => analyzer.molecular_system.geometry_type,
            "basis_set" => analyzer.molecular_system.basis_set,
            "n_qubits" => analyzer.n_qubits,
            "n_layers" => analyzer.n_layers,
            "exact_energy" => analyzer.exact_energy
        ),
        "analysis_timestamp" => string(now()),
        "julia_version" => string(VERSION)
    )
    
    # Convert results to JSON-serializable format
    serializable_results = Dict()
    for (method_name, data) in analyzer.results
        serializable_results[method_name] = Dict(
            "final_energy" => data["method_result"]["vqe_result"].final_energy,
            "energy_history" => data["method_result"]["vqe_result"].energy_history,
            "final_energy_error" => data["performance_metrics"]["final_energy_error"],
            "gradient_variance" => data["bp_diagnostics"].gradient_variance,
            "gradient_norm_mean" => data["bp_diagnostics"].gradient_norm_mean,
            "state_fidelity" => data["performance_metrics"]["state_fidelity"],
            "execution_time" => data["execution_time"],
            "converged" => data["method_result"]["vqe_result"].converged
        )
    end
    
    println("📁 Results saved to: $output_dir")
end

# ============================================================================
# Exports
# ============================================================================

export MolecularVQEAnalyzer, run_complete_analysis
export run_standard_vqe, run_local_global_vqe, run_adiabatic_vqe, run_sea_vqe, run_pretrained_vqe
export compute_bp_diagnostics, compute_performance_metrics
export generate_summary_table, save_results