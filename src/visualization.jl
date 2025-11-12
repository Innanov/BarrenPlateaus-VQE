"""
# Fixed Visualization Module (`visualization.jl`)

High-performance visualization tools for molecular VQE barren plateau analysis.
Updated for PDF output and grid layouts with all methods.

Key Features:
- Energy convergence plots (all methods in one plot)
- 3D loss landscape grid visualization 
- Contour landscape grid visualization
- Performance comparison tables (all methods)
- Export to PDF publication-ready formats
"""

# Import plotting packages with error handling
global PLOTTING_AVAILABLE = false
global PLOTS_MODULE = nothing

try
    import Plots
    global PLOTS_MODULE = Plots
    global PLOTTING_AVAILABLE = true
    
    # Try to set backend
    try
        import PlotlyJS
        Plots.plotlyjs()
        println("📊 Using PlotlyJS backend for interactive plots")
    catch
        try
            import GR
            Plots.gr()
            println("📊 Using GR backend for plots")
        catch
            println("📊 Using default backend for plots")
        end
    end
catch e
    @warn "Plots.jl not available: $e"
    global PLOTTING_AVAILABLE = false
end

# Import other dependencies
using DataFrames
using CSV
using Printf
using Statistics
using LinearAlgebra
using Random

# ============================================================================
# Loss Landscape Analysis
# ============================================================================

"""
    compute_loss_landscape_2d(cost_function, center_params::Vector{Float64};
                              param_indices::Tuple{Int,Int}=(1,2),
                              param_range::Float64=0.5,
                              resolution::Int=50)

Compute 2D loss landscape around a parameter point.
"""
function compute_loss_landscape_2d(cost_function, center_params::Vector{Float64};
                                  param_indices::Tuple{Int,Int}=(1,2),
                                  param_range::Float64=0.5,
                                  resolution::Int=50)
    
    i, j = param_indices
    if i > length(center_params) || j > length(center_params)
        throw(ArgumentError("Parameter indices out of bounds"))
    end
    
    # Create parameter grids
    param_i_range = range(center_params[i] - param_range, 
                         center_params[i] + param_range, 
                         length=resolution)
    param_j_range = range(center_params[j] - param_range,
                         center_params[j] + param_range,
                         length=resolution)
    
    # Compute loss landscape
    landscape = zeros(resolution, resolution)
    params_temp = copy(center_params)
    
    for (idx_i, val_i) in enumerate(param_i_range)
        for (idx_j, val_j) in enumerate(param_j_range)
            params_temp[i] = val_i
            params_temp[j] = val_j
            
            try
                landscape[idx_i, idx_j] = cost_function(params_temp)
            catch
                landscape[idx_i, idx_j] = NaN
            end
        end
    end
    
    return param_i_range, param_j_range, landscape
end

"""
    compute_trajectory_bounds(parameter_history::Vector{Vector{Float64}}, param_indices::Tuple{Int,Int})

Compute appropriate parameter range to capture the full optimization trajectory for loss landscape plotting.

This function analyzes the optimization trajectory in 2D parameter space and determines the bounds
needed to visualize the entire path. It adds a 20% margin beyond the trajectory bounds to ensure
the plot includes context around the optimization path.

# Arguments
- `parameter_history`: Vector of parameter vectors recorded during VQE optimization
- `param_indices`: Tuple (i, j) specifying which two parameters to analyze

# Returns
- `Float64`: The range value to use for plotting. Returns 0.5 as default if trajectory is empty.
  For non-empty trajectories, returns the maximum of the ranges for both parameters with 20% margin added.

# Example
```julia
# Compute bounds for parameters 1 and 2
range = compute_trajectory_bounds(vqe_result.parameter_history, (1, 2))
# Use range to create loss landscape plot centered on final parameters
```
"""
function compute_trajectory_bounds(parameter_history::Vector{Vector{Float64}}, param_indices::Tuple{Int,Int})
    if isempty(parameter_history)
        return 0.5  # Default range
    end
    
    i, j = param_indices
    trajectory_i, trajectory_j = compute_optimization_trajectory_2d(parameter_history; param_indices=param_indices)
    
    if isempty(trajectory_i)
        return 0.5
    end
    
    # Find bounds of trajectory
    min_i, max_i = extrema(trajectory_i)
    min_j, max_j = extrema(trajectory_j)
    
    # Add 20% margin beyond trajectory bounds
    range_i = max_i - min_i
    range_j = max_j - min_j
    margin = 0.2
    
    # Use the larger range dimension plus margin
    suggested_range = max(range_i, range_j) * (1 + margin)
    
    # Ensure minimum range for visualization
    return max(suggested_range, 0.3)
end
function compute_optimization_trajectory_2d(parameter_history::Vector{Vector{Float64}};
                                           param_indices::Tuple{Int,Int}=(1,2))
    i, j = param_indices
    
    if isempty(parameter_history)
        return Float64[], Float64[]
    end
    
    trajectory_i = [params[i] for params in parameter_history if length(params) > max(i,j)]
    trajectory_j = [params[j] for params in parameter_history if length(params) > max(i,j)]
    
    return trajectory_i, trajectory_j
end

# ============================================================================
# Plotting Helper Functions
# ============================================================================

"""
    check_plotting_available()

Check if plotting functionality is available.
"""
function check_plotting_available()
    if !PLOTTING_AVAILABLE
        @warn "Plotting not available. Install Plots.jl: using Pkg; Pkg.add(\"Plots\")"
        return false
    end
    return true
end

"""
    safe_plot(args...; kwargs...)

Safely create a plot, handling missing Plots.jl gracefully.
"""
function safe_plot(args...; kwargs...)
    if !check_plotting_available()
        return create_text_placeholder("Plots.jl not available")
    end
    
    try
        return PLOTS_MODULE.plot(args...; kwargs...)
    catch e
        @warn "Plot creation failed: $e"
        return create_text_placeholder("Plot creation failed")
    end
end

"""
    create_text_placeholder(message::String)

Create a text-based placeholder when plotting is not available or plot creation fails.

This function returns a dictionary representing a placeholder that can be displayed
when the plotting backend (Plots.jl or Makie.jl) is unavailable or when plot generation
encounters an error. The placeholder provides user-friendly feedback about why the plot
couldn't be created.

# Arguments
- `message`: String describing why the plot is unavailable (e.g., "Plotting backend not available")

# Returns
- `Dict{String, String}`: Dictionary with keys:
  - `"type"`: Always "text_placeholder"
  - `"message"`: The input message
  - `"display_text"`: Formatted message with emoji prefix "📊 {message}"

# Example
```julia
# When plotting fails
if !plotting_available()
    return create_text_placeholder("Plots.jl not installed")
end
```

# See Also
- [`check_plotting_available()`](@ref): Check if plotting backend is available
- [`safe_plot()`](@ref): Wrapper for safe plot creation with fallback
"""
function create_text_placeholder(message::String)
    return Dict(
        "type" => "text_placeholder",
        "message" => message,
        "display_text" => "📊 $message"
    )
end

"""
    create_molecule_subtitle(analyzer::MolecularVQEAnalyzer)

Create subtitle with molecule properties.
"""
function create_molecule_subtitle(analyzer::MolecularVQEAnalyzer)
    return "$(analyzer.molecular_system.name) | $(analyzer.molecular_system.geometry_type) | $(analyzer.molecular_system.basis_set) | $(analyzer.n_qubits) qubits | $(analyzer.n_layers) layers"
end

# ============================================================================
# All Methods Energy Convergence Visualization  
# ============================================================================

"""
    plot_all_methods_energy_convergence(analyzer::MolecularVQEAnalyzer; 
                                       save_path::Union{String, Nothing}=nothing)

Plot energy convergence for ALL VQE methods in a single plot.
"""
function plot_all_methods_energy_convergence(analyzer::MolecularVQEAnalyzer; 
                                            save_path::Union{String, Nothing}=nothing)
    
    if isempty(analyzer.results)
        @warn "No results to plot. Run analysis first."
        return create_placeholder_plot("No Results Available", "Run analysis first")
    end
    
    if !check_plotting_available()
        return create_text_placeholder("Plotting not available")
    end
    
    # Create main plot
    subtitle = create_molecule_subtitle(analyzer)
    p = safe_plot(size=(1200, 800), dpi=300, 
                 title="VQE Energy Convergence Comparison\n$(subtitle)",
                 titlefontsize=16,
                 xlabel="Iterations",
                 ylabel="Energy (Hartree)",
                 legend=:topright,
                 grid=true,
                 gridwidth=1,
                 gridcolor=:gray,
                 gridalpha=0.3,
                 left_margin=5PLOTS_MODULE.mm,
                 bottom_margin=5PLOTS_MODULE.mm)
    
    # Color palette and line styles for methods
    colors = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray, :cyan, :magenta]
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    
    method_count = 0
    valid_methods = []
    
    # Plot each method
    for (method_name, data) in analyzer.results
        if get(data["method_result"], "fallback", false)
            continue  # Skip fallback results
        end
        
        method_count += 1
        color = colors[((method_count-1) % length(colors)) + 1]
        line_style = line_styles[((method_count-1) % length(line_styles)) + 1]
        
        energies = data["method_result"]["vqe_result"].energy_history
        iterations = 1:length(energies)
        
        try
            PLOTS_MODULE.plot!(p, iterations, energies, 
                              label=replace(method_name, "_" => " "),
                              linewidth=3,
                              color=color,
                              linestyle=line_style,
                              alpha=0.8)
            push!(valid_methods, method_name)
        catch e
            @warn "Failed to add line for $method_name: $e"
        end
    end
    
    # Add exact ground state line
    try
        PLOTS_MODULE.hline!(p, [analyzer.exact_energy], 
                           label="Exact Ground State",
                           linestyle=:dash,
                           linewidth=3,
                           color=:black,
                           alpha=0.9)
    catch e
        @warn "Failed to add exact energy line: $e"
    end
    
    # Save if requested (PDF format)
    if save_path !== nothing && PLOTTING_AVAILABLE
        try
            PLOTS_MODULE.savefig(p, save_path)
            println("📊 Energy convergence plot saved to: $save_path")
        catch e
            @warn "Failed to save plot: $e"
        end
    end
    
    return p
end

# ============================================================================
# Grid Loss Landscape Visualization (3D)
# ============================================================================

"""
    plot_all_methods_loss_landscape_3d(analyzer::MolecularVQEAnalyzer;
                                       param_indices::Tuple{Int,Int}=(1,2),
                                       base_param_range::Float64=0.4,
                                       resolution::Int=50,
                                       save_path::Union{String, Nothing}=nothing)

Create individual publication-quality 3D loss landscape plots for each VQE method.
"""
function plot_all_methods_loss_landscape_3d(analyzer::MolecularVQEAnalyzer;
                                            param_indices::Tuple{Int,Int}=(1,2),
                                            base_param_range::Float64=0.4,
                                            resolution::Int=50,
                                            save_path::Union{String, Nothing}=nothing)
    
    if isempty(analyzer.results)
        @warn "No results to plot. Run analysis first."
        return create_text_placeholder("No results available")
    end
    
    if !check_plotting_available()
        return create_text_placeholder("3D plotting not available")
    end
    
    # Collect valid methods
    valid_methods = []
    for (method_name, data) in analyzer.results
        if !get(data["method_result"], "fallback", false)
            push!(valid_methods, (method_name, data))
        end
    end
    
    if isempty(valid_methods)
        return create_text_placeholder("No valid methods for landscape")
    end
    
    subtitle = create_molecule_subtitle(analyzer)
    saved_files = String[]
    
    println("🧮 Creating individual 3D loss landscapes for all methods...")
    
    for (i, (method_name, data)) in enumerate(valid_methods)
        try
            vqe_result = data["method_result"]["vqe_result"]
            ansatz = data["method_result"]["ansatz_info"]["ansatz"]
            final_params = vqe_result.final_parameters
            param_history = vqe_result.parameter_history
            
            if length(final_params) < max(param_indices...)
                @warn "Insufficient parameters for $method_name landscape"
                continue
            end
            
            # Compute adaptive parameter range to capture full trajectory
            adaptive_range = compute_trajectory_bounds(param_history, param_indices)
            actual_range = max(base_param_range, adaptive_range)
            
            println("  📊 $method_name: Using parameter range ±$(round(actual_range, digits=3))")
            
            # Create method-specific cost function
            function method_cost_function(params::Vector{Float64})
                return energy_evaluation(analyzer.hamiltonian, ansatz, params, analyzer.n_qubits)
            end
            
            # Compute landscape
            param_i_range, param_j_range, landscape = compute_loss_landscape_2d(
                method_cost_function, final_params;
                param_indices=param_indices,
                param_range=actual_range,
                resolution=resolution
            )
            
            # Create large individual 3D surface plot
            method_title = replace(method_name, "_" => " ")
            p_individual = PLOTS_MODULE.surface(collect(param_i_range), collect(param_j_range), landscape',
                                               title="$method_title VQE Loss Landscape\n$(subtitle)",
                                               titlefontsize=16,
                                               titlefont="Computer Modern",
                                               xlabel="θ$(param_indices[1])",
                                               ylabel="θ$(param_indices[2])",
                                               zlabel="Energy (Ha)",
                                               xlabelfontsize=14,
                                               ylabelfontsize=14,
                                               zlabelfontsize=14,
                                               xtickfontsize=12,
                                               ytickfontsize=12,
                                               ztickfontsize=12,
                                               camera=(45, 30),
                                               colorscale=:plasma,
                                               alpha=0.9,
                                               size=(900, 700),
                                               dpi=600,
                                               grid=true,
                                               gridwidth=1,
                                               gridalpha=0.3,
                                               background_color=:white,
                                               foreground_color=:black,
                                               left_margin=10PLOTS_MODULE.mm,
                                               right_margin=10PLOTS_MODULE.mm,
                                               top_margin=15PLOTS_MODULE.mm,
                                               bottom_margin=10PLOTS_MODULE.mm)
            
            # Add optimization trajectory
            if !isempty(param_history)
                trajectory_i, trajectory_j = compute_optimization_trajectory_2d(
                    param_history; param_indices=param_indices
                )
                
                if length(trajectory_i) > 1
                    # Compute trajectory energies
                    trajectory_energies = Float64[]
                    params_temp = copy(final_params)
                    
                    for k in 1:length(trajectory_i)
                        params_temp[param_indices[1]] = trajectory_i[k]
                        params_temp[param_indices[2]] = trajectory_j[k]
                        try
                            push!(trajectory_energies, method_cost_function(params_temp))
                        catch
                            push!(trajectory_energies, NaN)
                        end
                    end
                    
                    # Add trajectory line with multiple layers for visibility
                    PLOTS_MODULE.plot!(p_individual, trajectory_i, trajectory_j, trajectory_energies,
                                      linewidth=8,
                                      color=:black,
                                      alpha=0.8,
                                      label="")
                    PLOTS_MODULE.plot!(p_individual, trajectory_i, trajectory_j, trajectory_energies,
                                      linewidth=5,
                                      color=:white,
                                      alpha=1.0,
                                      label="")
                    PLOTS_MODULE.plot!(p_individual, trajectory_i, trajectory_j, trajectory_energies,
                                      linewidth=3,
                                      color=:yellow,
                                      alpha=1.0,
                                      label="Optimization Path")
                    
                    # Mark start point
                    if !isnan(trajectory_energies[1])
                        PLOTS_MODULE.scatter!(p_individual, [trajectory_i[1]], [trajectory_j[1]], [trajectory_energies[1]],
                                             markersize=15,
                                             color=:lime,
                                             label="Start",
                                             markerstrokewidth=3,
                                             markerstrokecolor=:black,
                                             marker=:circle)
                    end
                    
                    # Mark end point
                    if !isnan(trajectory_energies[end])
                        PLOTS_MODULE.scatter!(p_individual, [trajectory_i[end]], [trajectory_j[end]], [trajectory_energies[end]],
                                             markersize=15,
                                             color=:red,
                                             label="End",
                                             markerstrokewidth=3,
                                             markerstrokecolor=:black,
                                             marker=:star5)
                    end
                    
                    # Print trajectory statistics
                    traj_range_i = maximum(trajectory_i) - minimum(trajectory_i) 
                    traj_range_j = maximum(trajectory_j) - minimum(trajectory_j)
                    println("    Trajectory spans: θ$(param_indices[1])=±$(round(traj_range_i/2, digits=3)), θ$(param_indices[2])=±$(round(traj_range_j/2, digits=3))")
                end
            end
            
            # Save individual plot
            if save_path !== nothing
                # Extract directory and create individual filename
                save_dir = dirname(save_path)
                clean_name = replace(method_name, " " => "_", "/" => "_")
                individual_save_path = joinpath(save_dir, "landscape_3d_$(clean_name).pdf")
                
                try
                    PLOTS_MODULE.savefig(p_individual, individual_save_path)
                    push!(saved_files, individual_save_path)
                    println("    ✓ Saved: $(basename(individual_save_path))")
                catch e
                    @warn "Failed to save 3D plot for $method_name: $e"
                end
            end
            
        catch e
            @warn "Failed to create 3D landscape for $method_name: $e"
        end
    end
    
    if !isempty(saved_files)
        println("📊 Individual 3D loss landscapes saved:")
        for file in saved_files
            println("  📄 $(basename(file))")
        end
    end
    
    return saved_files
end

# ============================================================================
# Grid Loss Landscape Visualization (Contour)
# ============================================================================

"""
    plot_all_methods_loss_landscape_contour(analyzer::MolecularVQEAnalyzer;
                                            param_indices::Tuple{Int,Int}=(1,2),
                                            base_param_range::Float64=0.4,
                                            resolution::Int=80,
                                            save_path::Union{String, Nothing}=nothing)

Create individual publication-quality contour loss landscape plots for each VQE method.
"""
function plot_all_methods_loss_landscape_contour(analyzer::MolecularVQEAnalyzer;
                                                 param_indices::Tuple{Int,Int}=(1,2),
                                                 base_param_range::Float64=0.4,
                                                 resolution::Int=80,
                                                 save_path::Union{String, Nothing}=nothing)
    
    if isempty(analyzer.results)
        @warn "No results to plot. Run analysis first."
        return create_text_placeholder("No results available")
    end
    
    if !check_plotting_available()
        return create_text_placeholder("Contour plotting not available")
    end
    
    # Collect valid methods
    valid_methods = []
    for (method_name, data) in analyzer.results
        if !get(data["method_result"], "fallback", false)
            push!(valid_methods, (method_name, data))
        end
    end
    
    if isempty(valid_methods)
        return create_text_placeholder("No valid methods for landscape")
    end
    
    subtitle = create_molecule_subtitle(analyzer)
    saved_files = String[]
    
    println("🧮 Creating individual contour loss landscapes for all methods...")
    
    for (i, (method_name, data)) in enumerate(valid_methods)
        try
            vqe_result = data["method_result"]["vqe_result"]
            ansatz = data["method_result"]["ansatz_info"]["ansatz"]
            final_params = vqe_result.final_parameters
            param_history = vqe_result.parameter_history
            
            if length(final_params) < max(param_indices...)
                @warn "Insufficient parameters for $method_name landscape"
                continue
            end
            
            # Compute adaptive parameter range to capture full trajectory
            adaptive_range = compute_trajectory_bounds(param_history, param_indices)
            actual_range = max(base_param_range, adaptive_range)
            
            println("  📊 $method_name: Using parameter range ±$(round(actual_range, digits=3))")
            
            # Create method-specific cost function
            function method_cost_function(params::Vector{Float64})
                return energy_evaluation(analyzer.hamiltonian, ansatz, params, analyzer.n_qubits)
            end
            
            # Compute landscape
            param_i_range, param_j_range, landscape = compute_loss_landscape_2d(
                method_cost_function, final_params;
                param_indices=param_indices,
                param_range=actual_range,
                resolution=resolution
            )
            
            # Create large individual contour plot
            method_title = replace(method_name, "_" => " ")
            p_individual = PLOTS_MODULE.contour(collect(param_i_range), collect(param_j_range), landscape',
                                               title="$method_title VQE Loss Landscape\n$(subtitle)",
                                               titlefontsize=16,
                                               titlefont="Computer Modern",
                                               xlabel="θ$(param_indices[1])",
                                               ylabel="θ$(param_indices[2])",
                                               xlabelfontsize=14,
                                               ylabelfontsize=14,
                                               xtickfontsize=12,
                                               ytickfontsize=12,
                                               fill=true,
                                               colorscale=:plasma,
                                               levels=30,
                                               size=(1000, 700),
                                               dpi=600,
                                               aspect_ratio=:equal,
                                               grid=true,
                                               gridwidth=1,
                                               gridcolor=:gray,
                                               gridalpha=0.2,
                                               background_color=:white,
                                               foreground_color=:black,
                                               colorbar_title="",
                                               colorbar_titlefontsize=12,
                                               colorbar_tickfontsize=10,
                                               left_margin=15PLOTS_MODULE.mm,
                                               right_margin=25PLOTS_MODULE.mm,
                                               top_margin=15PLOTS_MODULE.mm,
                                               bottom_margin=15PLOTS_MODULE.mm)
            
            # Add optimization trajectory
            if !isempty(param_history)
                trajectory_i, trajectory_j = compute_optimization_trajectory_2d(
                    param_history; param_indices=param_indices
                )
                
                if length(trajectory_i) > 1
                    # Add trajectory line with multiple layers for maximum visibility
                    # Thick black outline
                    PLOTS_MODULE.plot!(p_individual, trajectory_i, trajectory_j,
                                      linewidth=8,
                                      color=:black,
                                      alpha=0.9,
                                      label="")
                    # White core line  
                    PLOTS_MODULE.plot!(p_individual, trajectory_i, trajectory_j,
                                      linewidth=5,
                                      color=:white,
                                      alpha=1.0,
                                      label="")
                    # Bright colored center line
                    PLOTS_MODULE.plot!(p_individual, trajectory_i, trajectory_j,
                                      linewidth=3,
                                      color=:yellow,
                                      alpha=1.0,
                                      label="Optimization Path")
                    
                    # Mark start point with enhanced visibility
                    PLOTS_MODULE.scatter!(p_individual, [trajectory_i[1]], [trajectory_j[1]],
                                         markersize=18,
                                         color=:lime,
                                         label="Start",
                                         markerstrokewidth=4,
                                         markerstrokecolor=:black,
                                         marker=:circle,
                                         markerstrokealpha=1.0,
                                         markeralpha=1.0)
                    
                    # Mark end point with enhanced visibility
                    PLOTS_MODULE.scatter!(p_individual, [trajectory_i[end]], [trajectory_j[end]],
                                         markersize=18,
                                         color=:red,
                                         label="End",
                                         markerstrokewidth=4,
                                         markerstrokecolor=:black,
                                         marker=:star5,
                                         markerstrokealpha=1.0,
                                         markeralpha=1.0)
                    
                    # Print trajectory statistics
                    traj_range_i = maximum(trajectory_i) - minimum(trajectory_i) 
                    traj_range_j = maximum(trajectory_j) - minimum(trajectory_j)
                    println("    Trajectory spans: θ$(param_indices[1])=±$(round(traj_range_i/2, digits=3)), θ$(param_indices[2])=±$(round(traj_range_j/2, digits=3))")
                end
            end
            
            # Save individual plot
            if save_path !== nothing
                # Extract directory and create individual filename
                save_dir = dirname(save_path)
                clean_name = replace(method_name, " " => "_", "/" => "_")
                individual_save_path = joinpath(save_dir, "landscape_contour_$(clean_name).pdf")
                
                try
                    PLOTS_MODULE.savefig(p_individual, individual_save_path)
                    push!(saved_files, individual_save_path)
                    println("    ✓ Saved: $(basename(individual_save_path))")
                catch e
                    @warn "Failed to save contour plot for $method_name: $e"
                end
            end
            
        catch e
            @warn "Failed to create contour landscape for $method_name: $e"
        end
    end
    
    if !isempty(saved_files)
        println("📊 Individual contour loss landscapes saved:")
        for file in saved_files
            println("  📄 $(basename(file))")
        end
    end
    
    return saved_files
end

# ============================================================================
# Performance Tables (All Methods - Fixed Printf Issues)
# ============================================================================

"""
    safe_sprintf(fmt::String, args...)

Safe sprintf function that handles format strings properly.
"""
function safe_sprintf(fmt::String, args...)
    try
        # Use @sprintf macro with literal format strings
        if fmt == "%-20s %-12s %-12s %-15s %-12s %-10s"
            return @sprintf("%-20s %-12s %-12s %-15s %-12s %-10s", args...)
        elseif fmt == "%-20s %-12.2e %-12.3f %-15.2e %-12.2e %-10.1f"
            return @sprintf("%-20s %-12.2e %-12.3f %-15.2e %-12.2e %-10.1f", args...)
        elseif fmt == "%.3f"
            return @sprintf("%.3f", args[1])
        elseif fmt == "%.1f"
            return @sprintf("%.1f", args[1])
        elseif fmt == "%.2f"
            return @sprintf("%.2f", args[1])
        elseif fmt == "%.0f"
            return @sprintf("%.0f", args[1])
        else
            # Fallback for other formats
            return string(args[1])
        end
    catch e
        @warn "sprintf failed: $e"
        return string(args[1])  # Simple fallback
    end
end

"""
    create_performance_table_all_methods(analyzer::MolecularVQEAnalyzer; 
                                        save_csv::Union{String, Nothing}=nothing,
                                        save_latex::Union{String, Nothing}=nothing)

Create performance comparison table including ALL methods with fixed Printf usage.
"""
function create_performance_table_all_methods(analyzer::MolecularVQEAnalyzer; 
                                             save_csv::Union{String, Nothing}=nothing,
                                             save_latex::Union{String, Nothing}=nothing)
    
    if isempty(analyzer.results)
        @warn "No results to create table. Run analysis first."
        return nothing
    end
    
    # Extract data for ALL methods (including fallback ones, but marked)
    methods = String[]
    energy_errors = Float64[]
    fidelities = Float64[]
    gradient_variances = Float64[]
    gradient_norms = Float64[]
    execution_times = Float64[]
    converged_flags = Bool[]
    final_energies = Float64[]
    
    for (method_name, data) in analyzer.results
        push!(methods, method_name)
        
        if get(data["method_result"], "fallback", false)
            # Fallback case - fill with placeholder values
            push!(energy_errors, NaN)
            push!(fidelities, NaN) 
            push!(gradient_variances, NaN)
            push!(gradient_norms, NaN)
            push!(execution_times, data["execution_time"])
            push!(converged_flags, false)
            push!(final_energies, NaN)
        else
            # Valid result
            push!(energy_errors, data["performance_metrics"]["final_energy_error"])
            push!(fidelities, data["performance_metrics"]["state_fidelity"])
            push!(gradient_variances, data["bp_diagnostics"].gradient_variance)
            push!(gradient_norms, data["bp_diagnostics"].gradient_norm_mean)
            push!(execution_times, data["execution_time"])
            push!(converged_flags, data["method_result"]["vqe_result"].converged)
            push!(final_energies, data["method_result"]["vqe_result"].final_energy)
        end
    end
    
    if isempty(methods)
        @warn "No methods found for table"
        return nothing
    end
    
    # Create DataFrame
    df = DataFrame(
        Method = methods,
        Final_Energy = final_energies,
        Energy_Error = energy_errors,
        State_Fidelity = fidelities,
        Gradient_Variance = gradient_variances,
        Gradient_Norm = gradient_norms,
        Execution_Time = execution_times,
        Converged = converged_flags
    )
    
    # Display formatted table
    println("\n📊 PERFORMANCE METRICS TABLE (ALL METHODS)")
    println("="^90)
    println("System: $(analyzer.molecular_system.name) ($(analyzer.molecular_system.geometry_type))")
    println("Qubits: $(analyzer.n_qubits), Layers: $(analyzer.n_layers)")
    println("Exact Energy: $(round(analyzer.exact_energy, digits=6)) Hartree")
    println("="^90)
    
    # Print formatted table using safe_sprintf
    header = safe_sprintf("%-20s %-12s %-12s %-12s %-15s %-12s %-10s %-10s", 
                         "Method", "Final Energy", "Energy Error", "Fidelity", "Grad Variance", "Grad Norm", "Time (s)", "Converged")
    println(header)
    println("-"^length(header))
    
    for i in 1:nrow(df)
        if isnan(df.Energy_Error[i])
            # Fallback method
            row = @sprintf("%-20s %-12s %-12s %-12s %-15s %-12s %-10.1f %-10s",
                          df.Method[i],
                          "FAILED",
                          "FAILED", 
                          "FAILED",
                          "FAILED",
                          "FAILED",
                          df.Execution_Time[i],
                          "No")
        else
            # Valid method
            row = @sprintf("%-20s %-12.6f %-12.2e %-12.3f %-15.2e %-12.2e %-10.1f %-10s",
                          df.Method[i],
                          df.Final_Energy[i],
                          df.Energy_Error[i],
                          df.State_Fidelity[i],
                          df.Gradient_Variance[i],
                          df.Gradient_Norm[i],
                          df.Execution_Time[i],
                          df.Converged[i] ? "Yes" : "No")
        end
        println(row)
    end
    println("="^90)
    
    # Save CSV if requested
    if save_csv !== nothing
        try
            CSV.write(save_csv, df)
            println("📁 CSV table saved to: $save_csv")
        catch e
            @warn "Failed to save CSV: $e"
        end
    end
    
    # Save LaTeX if requested
    if save_latex !== nothing
        try
            create_latex_table_all_methods(df, analyzer, save_latex)
        catch e
            @warn "Failed to save LaTeX table: $e"
        end
    end
    
    return df
end

"""
    create_latex_table_all_methods(df::DataFrame, analyzer::MolecularVQEAnalyzer, save_path::String)

Create LaTeX formatted table with all methods.
"""
function create_latex_table_all_methods(df::DataFrame, analyzer::MolecularVQEAnalyzer, save_path::String)
    latex_content = """
\\begin{table}[h!]
\\centering
\\caption{Performance Metrics for $(analyzer.molecular_system.name) - All VQE Methods ($(analyzer.molecular_system.geometry_type), $(analyzer.n_qubits) qubits)}
\\label{tab:performance_$(analyzer.molecular_system.name)_all_methods}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Method} & \\textbf{Final Energy} & \\textbf{Energy Error} & \\textbf{Fidelity} & \\textbf{Grad Variance} & \\textbf{Time (s)} & \\textbf{Converged} \\\\
\\hline
"""
    
    for i in 1:nrow(df)
        method = replace(df.Method[i], "_" => "\\_")
        
        if isnan(df.Energy_Error[i])
            # Fallback method
            latex_content *= "$method & \\textit{Failed} & \\textit{Failed} & \\textit{Failed} & \\textit{Failed} & $(round(df.Execution_Time[i], digits=1)) & No \\\\\n"
        else
            # Valid method
            final_energy = safe_sprintf("%.6f", df.Final_Energy[i])
            energy_error = format_scientific_latex(df.Energy_Error[i])
            fidelity = safe_sprintf("%.3f", df.State_Fidelity[i])
            grad_var = format_scientific_latex(df.Gradient_Variance[i])
            time_s = safe_sprintf("%.1f", df.Execution_Time[i])
            converged = df.Converged[i] ? "Yes" : "No"
            
            latex_content *= "$method & \$$(final_energy)\$ & \$$(energy_error)\$ & \$$(fidelity)\$ & \$$(grad_var)\$ & \$$(time_s)\$ & $(converged) \\\\\n"
        end
    end
    
    latex_content *= """
\\hline
\\end{tabular}
\\end{table}
"""
    
    open(save_path, "w") do f
        write(f, latex_content)
    end
    
    println("📁 LaTeX table saved to: $save_path")
end

"""
    format_scientific_latex(value::Float64; digits::Int=2)

Format a number for LaTeX scientific notation using safe sprintf.
"""
function format_scientific_latex(value::Float64; digits::Int=2)
    if isnan(value)
        return "\\text{NaN}"
    end
    
    if value == 0.0
        return "0"
    end
    
    exponent = floor(Int, log10(abs(value)))
    mantissa = value / (10.0^exponent)
    
    if abs(exponent) <= 3 && abs(value) >= 0.001
        # Use regular decimal notation
        return safe_sprintf("%.$(digits)f", value)
    else
        # Use scientific notation
        mantissa_str = safe_sprintf("%.$(digits-1)f", mantissa)
        return "$(mantissa_str) \\times 10^{$(exponent)}"
    end
end

# ============================================================================
# Legacy Functions (Updated for PDF and molecule info)
# ============================================================================

"""
    plot_energy_convergence(analyzer::MolecularVQEAnalyzer; 
                           save_path::Union{String, Nothing}=nothing,
                           show_exact::Bool=true,
                           title_override::Union{String, Nothing}=nothing,
                           log_scale::Bool=false)

Plot energy convergence (legacy function, redirects to all methods version).
"""
function plot_energy_convergence(analyzer::MolecularVQEAnalyzer; 
                                save_path::Union{String, Nothing}=nothing,
                                show_exact::Bool=true,
                                title_override::Union{String, Nothing}=nothing,
                                log_scale::Bool=false)
    
    # Redirect to all methods version (ignore log_scale as requested)
    return plot_all_methods_energy_convergence(analyzer; save_path=save_path)
end

"""
    plot_loss_landscape_3d(cost_function, center_params::Vector{Float64};
                           param_indices::Tuple{Int,Int}=(1,2),
                           param_range::Float64=0.5,
                           resolution::Int=30,
                           show_trajectory::Bool=false,
                           parameter_history::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                           save_path::Union{String, Nothing}=nothing)

Create 3D loss landscape visualization (legacy, now saves as PDF).
"""
function plot_loss_landscape_3d(cost_function, center_params::Vector{Float64};
                                param_indices::Tuple{Int,Int}=(1,2),
                                param_range::Float64=0.5,
                                resolution::Int=30,
                                show_trajectory::Bool=false,
                                parameter_history::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                                save_path::Union{String, Nothing}=nothing)
    
    if !check_plotting_available()
        return create_text_placeholder("3D plotting not available")
    end
    
    println("🧮 Computing 3D loss landscape...")
    param_i_range, param_j_range, landscape = compute_loss_landscape_2d(
        cost_function, center_params;
        param_indices=param_indices,
        param_range=param_range,
        resolution=resolution
    )
    
    # Create surface plot
    try
        p = PLOTS_MODULE.surface(collect(param_i_range), collect(param_j_range), landscape',
                                title="VQE Loss Landscape",
                                xlabel="Parameter $(param_indices[1])",
                                ylabel="Parameter $(param_indices[2])",
                                zlabel="Energy",
                                camera=(45, 30),
                                size=(800, 600),
                                dpi=300,
                                colorscale=:viridis,
                                alpha=0.8)
        
        # Add optimization trajectory if requested
        if show_trajectory && parameter_history !== nothing
            trajectory_i, trajectory_j = compute_optimization_trajectory_2d(
                parameter_history; param_indices=param_indices
            )
            
            if length(trajectory_i) > 1
                # Compute trajectory energies
                trajectory_energies = Float64[]
                params_temp = copy(center_params)
                
                for k in 1:length(trajectory_i)
                    params_temp[param_indices[1]] = trajectory_i[k]
                    params_temp[param_indices[2]] = trajectory_j[k]
                    try
                        push!(trajectory_energies, cost_function(params_temp))
                    catch
                        push!(trajectory_energies, NaN)
                    end
                end
                
                # Add trajectory line
                PLOTS_MODULE.plot!(p, trajectory_i, trajectory_j, trajectory_energies,
                                  linewidth=4,
                                  color=:red,
                                  alpha=0.9,
                                  label="Optimization Path")
                
                # Mark start and end points
                if !isnan(trajectory_energies[1])
                    PLOTS_MODULE.scatter!(p, [trajectory_i[1]], [trajectory_j[1]], [trajectory_energies[1]],
                                         markersize=8,
                                         color=:green,
                                         label="Start",
                                         markerstrokewidth=2,
                                         markerstrokecolor=:white)
                end
                
                if !isnan(trajectory_energies[end])
                    PLOTS_MODULE.scatter!(p, [trajectory_i[end]], [trajectory_j[end]], [trajectory_energies[end]],
                                         markersize=8,
                                         color=:red,
                                         label="End",
                                         markerstrokewidth=2,
                                         markerstrokecolor=:white)
                end
            end
        end
        
        # Save if requested (PDF format)
        if save_path !== nothing
            try
                PLOTS_MODULE.savefig(p, save_path)
                println("📊 3D loss landscape saved to: $save_path")
            catch e
                @warn "Failed to save 3D plot: $e"
            end
        end
        
        return p
        
    catch e
        @warn "Failed to create 3D surface plot: $e"
        return create_text_placeholder("3D surface plot creation failed")
    end
end

"""
    plot_loss_landscape_contour(cost_function, center_params::Vector{Float64};
                                param_indices::Tuple{Int,Int}=(1,2),
                                param_range::Float64=0.5,
                                resolution::Int=50,
                                show_trajectory::Bool=false,
                                parameter_history::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                                save_path::Union{String, Nothing}=nothing)

Create 2D contour plot (legacy, now saves as PDF).
"""
function plot_loss_landscape_contour(cost_function, center_params::Vector{Float64};
                                     param_indices::Tuple{Int,Int}=(1,2),
                                     param_range::Float64=0.5,
                                     resolution::Int=50,
                                     show_trajectory::Bool=false,
                                     parameter_history::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                                     save_path::Union{String, Nothing}=nothing)
    
    if !check_plotting_available()
        return create_text_placeholder("Contour plotting not available")
    end
    
    println("🧮 Computing 2D contour landscape...")
    param_i_range, param_j_range, landscape = compute_loss_landscape_2d(
        cost_function, center_params;
        param_indices=param_indices,
        param_range=param_range,
        resolution=resolution
    )
    
    try
        # Create contour plot
        p = PLOTS_MODULE.contour(collect(param_i_range), collect(param_j_range), landscape',
                                title="VQE Loss Landscape (Contour View)",
                                xlabel="Parameter $(param_indices[1])",
                                ylabel="Parameter $(param_indices[2])",
                                size=(800, 600),
                                dpi=300,
                                fill=true,
                                colorscale=:viridis,
                                levels=20)
        
        # Add optimization trajectory if requested
        if show_trajectory && parameter_history !== nothing
            trajectory_i, trajectory_j = compute_optimization_trajectory_2d(
                parameter_history; param_indices=param_indices
            )
            
            if length(trajectory_i) > 1
                # Add trajectory line
                PLOTS_MODULE.plot!(p, trajectory_i, trajectory_j,
                                  linewidth=3,
                                  color=:red,
                                  alpha=0.9,
                                  label="Optimization Path")
                
                # Mark start and end points
                PLOTS_MODULE.scatter!(p, [trajectory_i[1]], [trajectory_j[1]],
                                     markersize=8,
                                     color=:green,
                                     label="Start",
                                     markerstrokewidth=2,
                                     markerstrokecolor=:white)
                
                PLOTS_MODULE.scatter!(p, [trajectory_i[end]], [trajectory_j[end]],
                                     markersize=8,
                                     color=:red,
                                     label="End",
                                     markerstrokewidth=2,
                                     markerstrokecolor=:white)
            end
        end
        
        # Save if requested (PDF format)
        if save_path !== nothing
            try
                PLOTS_MODULE.savefig(p, save_path)
                println("📊 Loss landscape contour saved to: $save_path")
            catch e
                @warn "Failed to save contour plot: $e"
            end
        end
        
        return p
        
    catch e
        @warn "Failed to create contour plot: $e"
        return create_text_placeholder("Contour plot creation failed")
    end
end

# ============================================================================
# Updated Quick Analysis Plot (No Bar Charts)
# ============================================================================

"""
    quick_analysis_plot(analyzer::MolecularVQEAnalyzer)

Generate enhanced quick summary plot (no bar charts, just energy convergence).
"""
function quick_analysis_plot(analyzer::MolecularVQEAnalyzer)
    if isempty(analyzer.results)
        # Create a demonstration plot showing system info
        println("📊 Creating demonstration plot (no analysis results yet)")
        
        if !check_plotting_available()
            return create_text_placeholder("System ready: $(analyzer.molecular_system.name), $(analyzer.n_qubits) qubits")
        end
        
        try
            p = safe_plot(xlims=(0, 10), ylims=(0, 10),
                         title="BarrenPlateausVQE.jl Ready",
                         legend=false,
                         grid=false,
                         showaxis=false,
                         size=(700, 500))
            
            PLOTS_MODULE.annotate!(p, 5, 8, PLOTS_MODULE.text("System: $(analyzer.molecular_system.name)", :center, 16))
            PLOTS_MODULE.annotate!(p, 5, 7, PLOTS_MODULE.text("Qubits: $(analyzer.n_qubits)", :center, 14))
            PLOTS_MODULE.annotate!(p, 5, 6, PLOTS_MODULE.text("Layers: $(analyzer.n_layers)", :center, 14))
            PLOTS_MODULE.annotate!(p, 5, 4, PLOTS_MODULE.text("Ready for Analysis!", :center, 18, :green))
            PLOTS_MODULE.annotate!(p, 5, 2, PLOTS_MODULE.text("Run: results = run_complete_analysis(analyzer)", :center, 12, :gray))
            
            return p
        catch e
            return create_text_placeholder("Ready: $(analyzer.molecular_system.name), $(analyzer.n_qubits) qubits")
        end
    end
    
    # Just return energy convergence plot (no bar charts)
    return plot_all_methods_energy_convergence(analyzer)
end

# ============================================================================
# Placeholder Functions
# ============================================================================

"""
    create_placeholder_plot(title::String, message::String)

Create a placeholder plot when no data is available.
"""
function create_placeholder_plot(title::String, message::String)
    if !check_plotting_available()
        return create_text_placeholder("$title: $message")
    end
    
    try
        p = safe_plot(xlims=(0, 1), ylims=(0, 1), 
                     title=title,
                     legend=false,
                     grid=false,
                     showaxis=false,
                     size=(600, 400))
        
        PLOTS_MODULE.annotate!(p, 0.5, 0.5, PLOTS_MODULE.text(message, :center, 14))
        PLOTS_MODULE.annotate!(p, 0.5, 0.3, PLOTS_MODULE.text("Run analyzer.run_complete_analysis() first", :center, 10, :gray))
        
        return p
    catch e
        return create_text_placeholder("$title: $message")
    end
end

# ============================================================================
# Comprehensive Visualization Pipeline (Updated)
# ============================================================================

"""
    create_comprehensive_visualization(analyzer::MolecularVQEAnalyzer;
                                      output_dir::String="./plots",
                                      param_indices::Tuple{Int,Int}=(1,2),
                                      landscape_resolution::Int=50)

Create comprehensive publication-quality visualization including all methods (PDF format).
"""
function create_comprehensive_visualization(analyzer::MolecularVQEAnalyzer;
                                           output_dir::String="./plots",
                                           param_indices::Tuple{Int,Int}=(1,2),
                                           landscape_resolution::Int=50)
    
    if isempty(analyzer.results)
        @warn "No results to visualize. Run analysis first."
        return
    end
    
    if !check_plotting_available()
        println("📊 Plotting not available, creating text summaries instead")
        return
    end
    
    # Create output directory
    mkpath(output_dir)
    println("📊 Creating comprehensive visualization suite...")
    
    # 1. All methods energy convergence plot
    println("  📊 All methods energy convergence plot...")
    try
        conv_plot = plot_all_methods_energy_convergence(analyzer; 
                                                       save_path=joinpath(output_dir, "energy_convergence_all_methods.pdf"))
        println("    ✓ All methods energy convergence plot saved")
    catch e
        @warn "Energy convergence plot failed: $e"
    end
    
    # 2. 3D loss landscape individual plots
    println("  📊 Individual 3D loss landscape plots...")
    try
        landscape_3d_files = plot_all_methods_loss_landscape_3d(analyzer;
                                                               param_indices=param_indices,
                                                               base_param_range=0.5,
                                                               resolution=landscape_resolution,
                                                               save_path=joinpath(output_dir, "dummy_3d.pdf"))
        println("    ✓ Individual 3D loss landscape plots saved")
    catch e
        @warn "3D landscape plots failed: $e"
    end
    
    # 3. Contour loss landscape individual plots
    println("  📊 Individual contour loss landscape plots...")
    try
        landscape_contour_files = plot_all_methods_loss_landscape_contour(analyzer;
                                                                         param_indices=param_indices,
                                                                         base_param_range=0.5,
                                                                         resolution=80,
                                                                         save_path=joinpath(output_dir, "dummy_contour.pdf"))
        println("    ✓ Individual contour loss landscape plots saved")
    catch e
        @warn "Contour landscape plots failed: $e"
    end
    
    # 4. Performance comparison table (all methods)
    println("  📊 Performance table (all methods)...")
    try
        df = create_performance_table_all_methods(analyzer, 
                                                 save_csv=joinpath(output_dir, "performance_table_all_methods.csv"))
        if df !== nothing
            println("    ✓ Performance table saved")
        end
    catch e
        @warn "Performance table failed: $e"
    end
    
    println("📊 Comprehensive visualization completed!")
    println("   Files saved to: $output_dir")
end

# ============================================================================
# Stub Functions for Compatibility
# ============================================================================

"""
    plot_gradient_diagnostics(analyzer::MolecularVQEAnalyzer; kwargs...)

Stub function for gradient diagnostics plotting.
"""
function plot_gradient_diagnostics(analyzer::MolecularVQEAnalyzer; kwargs...)
    if !check_plotting_available()
        return create_text_placeholder("Gradient diagnostics plotting not available")
    end
    
    return create_text_placeholder("Gradient diagnostics not implemented yet")
end

"""
    plot_method_comparison(analyzer::MolecularVQEAnalyzer; kwargs...)

Stub function for method comparison plotting.
"""
function plot_method_comparison(analyzer::MolecularVQEAnalyzer; kwargs...)
    if !check_plotting_available()
        return create_text_placeholder("Method comparison plotting not available")
    end
    
    return create_text_placeholder("Method comparison not implemented yet")
end

"""
    generate_all_plots(analyzer::MolecularVQEAnalyzer; kwargs...)

Generate all available plots (PDF format).
"""
function generate_all_plots(analyzer::MolecularVQEAnalyzer; kwargs...)
    output_dir = get(kwargs, :output_dir, "./plots")
    
    if !check_plotting_available()
        @warn "Plotting not available. Install Plots.jl to enable visualization."
        return String[]
    end
    
    try
        create_comprehensive_visualization(analyzer; output_dir=output_dir)
        return readdir(output_dir)
    catch e
        @warn "Plot generation failed: $e"
        return String[]
    end
end

"""
    save_analysis_summary(analyzer::MolecularVQEAnalyzer, output_dir::String)

Save a comprehensive analysis summary.
"""
function save_analysis_summary(analyzer::MolecularVQEAnalyzer, output_dir::String)
    mkpath(output_dir)
    
    # Create summary report
    summary_path = joinpath(output_dir, "analysis_summary.md")
    
    open(summary_path, "w") do f
        write(f, "# VQE Barren Plateau Analysis Summary\n\n")
        write(f, "## System Information\n")
        write(f, "- **Molecule**: $(analyzer.molecular_system.name)\n")
        write(f, "- **Geometry**: $(analyzer.molecular_system.geometry_type)\n")
        write(f, "- **Basis Set**: $(analyzer.molecular_system.basis_set)\n")
        write(f, "- **Qubits**: $(analyzer.n_qubits)\n")
        write(f, "- **Layers**: $(analyzer.n_layers)\n")
        write(f, "- **Exact Energy**: $(analyzer.exact_energy)\n\n")
        
        write(f, "## Results Summary\n\n")
        
        if isempty(analyzer.results)
            write(f, "No analysis results available. Run `run_complete_analysis(analyzer)` first.\n")
        else
            for (method_name, data) in analyzer.results
                write(f, "### $method_name\n")
                
                if get(data["method_result"], "fallback", false)
                    write(f, "- **Status**: FAILED (fallback)\n")
                    write(f, "- **Execution Time**: $(round(data["execution_time"], digits=2))s\n\n")
                else
                    write(f, "- **Final Energy**: $(data["method_result"]["vqe_result"].final_energy)\n")
                    write(f, "- **Energy Error**: $(data["performance_metrics"]["final_energy_error"])\n")
                    write(f, "- **Gradient Variance**: $(data["bp_diagnostics"].gradient_variance)\n")
                    write(f, "- **State Fidelity**: $(data["performance_metrics"]["state_fidelity"])\n")
                    write(f, "- **Execution Time**: $(round(data["execution_time"], digits=2))s\n")
                    write(f, "- **Converged**: $(data["method_result"]["vqe_result"].converged)\n\n")
                end
            end
        end
    end
    
    println("📄 Analysis summary saved to: $summary_path")
end

# ============================================================================
# Exports
# ============================================================================

export plot_energy_convergence, plot_loss_landscape_3d, plot_loss_landscape_contour
export plot_all_methods_energy_convergence, plot_all_methods_loss_landscape_3d, plot_all_methods_loss_landscape_contour
export create_comprehensive_visualization, quick_analysis_plot
export create_performance_table_all_methods, create_latex_table_all_methods
export compute_loss_landscape_2d, compute_optimization_trajectory_2d
export plot_gradient_diagnostics, plot_method_comparison, generate_all_plots
export save_analysis_summary