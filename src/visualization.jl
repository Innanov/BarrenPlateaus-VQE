"""
# Fixed Visualization Module (`visualization.jl`)

High-performance visualization tools for molecular VQE barren plateau analysis.
Fixed Printf.sprintf issues.

Key Features:
- Energy convergence plots
- 3D loss landscape visualization
- Optimization trajectory visualization
- Performance comparison tables
- Export to publication-ready formats
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
    compute_optimization_trajectory_2d(parameter_history::Vector{Vector{Float64}};
                                      param_indices::Tuple{Int,Int}=(1,2))

Extract 2D trajectory from parameter history.
"""
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

Create a text-based placeholder when plotting is not available.
"""
function create_text_placeholder(message::String)
    return Dict(
        "type" => "text_placeholder",
        "message" => message,
        "display_text" => "📊 $message"
    )
end

# ============================================================================
# Energy Convergence Visualization
# ============================================================================

"""
    plot_energy_convergence(analyzer::MolecularVQEAnalyzer; 
                           save_path::Union{String, Nothing}=nothing,
                           show_exact::Bool=true,
                           title_override::Union{String, Nothing}=nothing,
                           log_scale::Bool=false)

Plot energy convergence for all VQE methods with enhanced visualization.
"""
function plot_energy_convergence(analyzer::MolecularVQEAnalyzer; 
                                save_path::Union{String, Nothing}=nothing,
                                show_exact::Bool=true,
                                title_override::Union{String, Nothing}=nothing,
                                log_scale::Bool=false)
    
    if isempty(analyzer.results)
        @warn "No results to plot. Run analysis first."
        return create_placeholder_plot("No Results Available", "Run analysis first")
    end
    
    if !check_plotting_available()
        return create_text_placeholder("Plotting not available")
    end
    
    # Create main plot
    p = safe_plot(size=(1000, 700), dpi=300, 
                 title=title_override !== nothing ? title_override : 
                       "Energy Convergence: $(analyzer.molecular_system.name)",
                 xlabel="Iterations",
                 ylabel=log_scale ? "log₁₀|Energy - Exact|" : "Energy",
                 legend=:topright,
                 grid=true,
                 gridwidth=1,
                 gridcolor=:gray,
                 gridalpha=0.3)
    
    # Color palette for methods
    colors = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray]
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
        
        if log_scale && show_exact
            # Plot log scale of energy error
            energy_errors = abs.(energies .- analyzer.exact_energy)
            # Avoid log(0) by adding small epsilon
            energy_errors = max.(energy_errors, 1e-12)
            plot_energies = log10.(energy_errors)
        else
            plot_energies = energies
        end
        
        try
            PLOTS_MODULE.plot!(p, iterations, plot_energies, 
                              label=method_name,
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
    if show_exact && !log_scale
        try
            PLOTS_MODULE.hline!(p, [analyzer.exact_energy], 
                               label="Exact Ground State",
                               linestyle=:dash,
                               linewidth=2,
                               color=:black,
                               alpha=0.9)
        catch e
            @warn "Failed to add exact energy line: $e"
        end
    end
    
    # Save if requested
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
# 3D Loss Landscape Visualization
# ============================================================================

"""
    plot_loss_landscape_3d(cost_function, center_params::Vector{Float64};
                           param_indices::Tuple{Int,Int}=(1,2),
                           param_range::Float64=0.5,
                           resolution::Int=30,
                           show_trajectory::Bool=false,
                           parameter_history::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                           save_path::Union{String, Nothing}=nothing)

Create 3D loss landscape visualization with optional optimization trajectory.
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
        
        # Save if requested
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

Create 2D contour plot of loss landscape with optimization trajectory.
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
        
        # Save if requested
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
# Quick Analysis Plot (Enhanced)
# ============================================================================

"""
    quick_analysis_plot(analyzer::MolecularVQEAnalyzer)

Generate enhanced quick summary plot for immediate analysis.
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
    
    # Create enhanced combined plot with results
    if !check_plotting_available()
        return create_text_placeholder("Analysis complete - plotting not available")
    end
    
    try
        # Create subplots
        p1 = plot_energy_convergence(analyzer; show_exact=true)
        
        # Performance comparison bar chart
        methods = String[]
        final_energies = Float64[]
        energy_errors = Float64[]
        gradient_vars = Float64[]
        
        for (method_name, data) in analyzer.results
            if !get(data["method_result"], "fallback", false)
                push!(methods, method_name)
                push!(final_energies, data["method_result"]["vqe_result"].final_energy)
                push!(energy_errors, data["performance_metrics"]["final_energy_error"])
                push!(gradient_vars, data["bp_diagnostics"].gradient_variance)
            end
        end
        
        if !isempty(methods)
            # Energy comparison
            p2 = PLOTS_MODULE.bar(methods, final_energies,
                                 title="Final Energies",
                                 ylabel="Energy",
                                 xrotation=45,
                                 color=:steelblue,
                                 alpha=0.7,
                                 legend=false)
            
            # Add exact energy line
            try
                PLOTS_MODULE.hline!(p2, [analyzer.exact_energy], 
                                   linestyle=:dash,
                                   color=:red,
                                   linewidth=2,
                                   label="Exact")
            catch
                # Continue without exact line if it fails
            end
            
            # Error comparison (log scale)
            p3 = PLOTS_MODULE.bar(methods, log10.(energy_errors),
                                 title="Energy Errors (log₁₀)",
                                 ylabel="log₁₀(Energy Error)",
                                 xrotation=45,
                                 color=:coral,
                                 alpha=0.7,
                                 legend=false)
            
            # Gradient variance comparison
            p4 = PLOTS_MODULE.bar(methods, log10.(gradient_vars),
                                 title="Gradient Variance (log₁₀)",
                                 ylabel="log₁₀(Gradient Variance)",
                                 xrotation=45,
                                 color=:lightgreen,
                                 alpha=0.7,
                                 legend=false)
            
            combined = PLOTS_MODULE.plot(p1, p2, p3, p4, 
                                        layout=(2, 2), 
                                        size=(1200, 900),
                                        plot_title="VQE Analysis: $(analyzer.molecular_system.name)")
        else
            combined = p1
        end
        
        return combined
        
    catch e
        @warn "Failed to create quick analysis plot: $e"
        return create_text_placeholder("Analysis complete - plot creation failed")
    end
end

# ============================================================================
# Performance Tables (Fixed Printf Issues)
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
    create_performance_table(analyzer::MolecularVQEAnalyzer; 
                            save_csv::Union{String, Nothing}=nothing,
                            save_latex::Union{String, Nothing}=nothing)

Create performance comparison table with fixed Printf usage.
"""
function create_performance_table(analyzer::MolecularVQEAnalyzer; 
                                 save_csv::Union{String, Nothing}=nothing,
                                 save_latex::Union{String, Nothing}=nothing)
    
    if isempty(analyzer.results)
        @warn "No results to create table. Run analysis first."
        return nothing
    end
    
    # Extract data
    methods = String[]
    energy_errors = Float64[]
    fidelities = Float64[]
    gradient_variances = Float64[]
    gradient_norms = Float64[]
    execution_times = Float64[]
    converged_flags = Bool[]
    
    for (method_name, data) in analyzer.results
        if get(data["method_result"], "fallback", false)
            continue  # Skip fallback results in table
        end
        
        push!(methods, method_name)
        push!(energy_errors, data["performance_metrics"]["final_energy_error"])
        push!(fidelities, data["performance_metrics"]["state_fidelity"])
        push!(gradient_variances, data["bp_diagnostics"].gradient_variance)
        push!(gradient_norms, data["bp_diagnostics"].gradient_norm_mean)
        push!(execution_times, data["execution_time"])
        push!(converged_flags, data["method_result"]["vqe_result"].converged)
    end
    
    if isempty(methods)
        @warn "No valid results for table"
        return nothing
    end
    
    # Create DataFrame
    df = DataFrame(
        Method = methods,
        Energy_Error = energy_errors,
        State_Fidelity = fidelities,
        Gradient_Variance = gradient_variances,
        Gradient_Norm = gradient_norms,
        Execution_Time = execution_times,
        Converged = converged_flags
    )
    
    # Display formatted table
    println("\n📊 PERFORMANCE METRICS TABLE")
    println("="^80)
    println("System: $(analyzer.molecular_system.name) ($(analyzer.molecular_system.geometry_type))")
    println("Qubits: $(analyzer.n_qubits), Layers: $(analyzer.n_layers)")
    println("="^80)
    
    # Print formatted table using safe_sprintf
    header = safe_sprintf("%-20s %-12s %-12s %-15s %-12s %-10s", 
                         "Method", "Energy Error", "Fidelity", "Grad Variance", "Grad Norm", "Time (s)")
    println(header)
    println("-"^length(header))
    
    for i in 1:nrow(df)
        row = safe_sprintf("%-20s %-12.2e %-12.3f %-15.2e %-12.2e %-10.1f",
                          df.Method[i],
                          df.Energy_Error[i],
                          df.State_Fidelity[i],
                          df.Gradient_Variance[i],
                          df.Gradient_Norm[i],
                          df.Execution_Time[i])
        println(row)
    end
    println("="^80)
    
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
            create_latex_table(df, analyzer, save_latex)
        catch e
            @warn "Failed to save LaTeX table: $e"
        end
    end
    
    return df
end

"""
    create_latex_table(df::DataFrame, analyzer::MolecularVQEAnalyzer, save_path::String)

Create LaTeX formatted table with fixed Printf usage.
"""
function create_latex_table(df::DataFrame, analyzer::MolecularVQEAnalyzer, save_path::String)
    latex_content = """
\\begin{table}[h!]
\\centering
\\caption{Performance Metrics for $(analyzer.molecular_system.name) ($(analyzer.molecular_system.geometry_type), $(analyzer.n_qubits) qubits)}
\\label{tab:performance_$(analyzer.molecular_system.name)_$(analyzer.molecular_system.geometry_type)}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Method} & \\textbf{Energy Error} & \\textbf{State Fidelity} & \\textbf{Gradient Variance} & \\textbf{Time (s)} \\\\
\\hline
"""
    
    for i in 1:nrow(df)
        method = replace(df.Method[i], "_" => "\\_")
        energy_error = format_scientific_latex(df.Energy_Error[i])
        fidelity = safe_sprintf("%.3f", df.State_Fidelity[i])
        grad_var = format_scientific_latex(df.Gradient_Variance[i])
        time_s = safe_sprintf("%.1f", df.Execution_Time[i])
        
        latex_content *= "$method & \$$(energy_error)\$ & \$$(fidelity)\$ & \$$(grad_var)\$ & \$$(time_s)\$ \\\\\n"
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
# Comprehensive Visualization Pipeline
# ============================================================================

"""
    create_comprehensive_visualization(analyzer::MolecularVQEAnalyzer;
                                      output_dir::String="./plots",
                                      param_indices::Tuple{Int,Int}=(1,2),
                                      landscape_resolution::Int=30)

Create comprehensive visualization including energy convergence and loss landscapes.
"""
function create_comprehensive_visualization(analyzer::MolecularVQEAnalyzer;
                                           output_dir::String="./plots",
                                           param_indices::Tuple{Int,Int}=(1,2),
                                           landscape_resolution::Int=30)
    
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
    
    # 1. Energy convergence plot
    println("  📊 Energy convergence plot...")
    try
        conv_plot = plot_energy_convergence(analyzer; 
                                           save_path=joinpath(output_dir, "energy_convergence.png"))
        println("    ✓ Energy convergence plot saved")
    catch e
        @warn "Energy convergence plot failed: $e"
    end
    
    # 2. Log-scale energy convergence plot
    println("  📊 Log-scale energy convergence plot...")
    try
        conv_log_plot = plot_energy_convergence(analyzer; 
                                               log_scale=true,
                                               save_path=joinpath(output_dir, "energy_convergence_log.png"))
        println("    ✓ Log-scale convergence plot saved")
    catch e
        @warn "Log-scale convergence plot failed: $e"
    end
    
    # 3. Performance comparison table
    println("  📊 Performance table...")
    try
        df = create_performance_table(analyzer, 
                                     save_csv=joinpath(output_dir, "performance_table.csv"))
        if df !== nothing
            println("    ✓ Performance table saved")
        end
    catch e
        @warn "Performance table failed: $e"
    end
    
    # 4. Method-specific loss landscapes (limit to 2 for performance)
    landscape_count = 0
    max_landscapes = 2
    
    for (method_name, data) in analyzer.results
        if get(data["method_result"], "fallback", false) || landscape_count >= max_landscapes
            continue
        end
        
        try
            vqe_result = data["method_result"]["vqe_result"]
            ansatz = data["method_result"]["ansatz_info"]["ansatz"]
            final_params = vqe_result.final_parameters
            param_history = vqe_result.parameter_history
            
            if length(final_params) >= max(param_indices...)
                println("  📊 Creating landscapes for $method_name...")
                
                # Create method-specific cost function
                function method_cost_function(params::Vector{Float64})
                    return energy_evaluation(analyzer.hamiltonian, ansatz, params, analyzer.n_qubits)
                end
                
                # Clean method name for filename
                clean_name = replace(method_name, " " => "_", "/" => "_")
                
                # 3D landscape
                try
                    plot_loss_landscape_3d(
                        method_cost_function, final_params;
                        param_indices=param_indices,
                        param_range=0.3,
                        resolution=landscape_resolution,
                        show_trajectory=true,
                        parameter_history=param_history,
                        save_path=joinpath(output_dir, "landscape_3d_$(clean_name).png")
                    )
                    println("    ✓ 3D landscape saved")
                catch e
                    @warn "3D landscape failed for $method_name: $e"
                end
                
                # Contour landscape
                try
                    plot_loss_landscape_contour(
                        method_cost_function, final_params;
                        param_indices=param_indices,
                        param_range=0.3,
                        resolution=40,
                        show_trajectory=true,
                        parameter_history=param_history,
                        save_path=joinpath(output_dir, "landscape_contour_$(clean_name).png")
                    )
                    println("    ✓ Contour landscape saved")
                catch e
                    @warn "Contour landscape failed for $method_name: $e"
                end
                
                landscape_count += 1
                
            else
                println("    ⚠️  Insufficient parameters for $method_name landscape")
            end
            
        catch e
            @warn "Landscape creation failed for $method_name: $e"
        end
    end
    
    println("📊 Comprehensive visualization completed!")
    println("   Files saved to: $output_dir")
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

Generate all available plots.
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
                if get(data["method_result"], "fallback", false)
                    continue
                end
                
                write(f, "### $method_name\n")
                write(f, "- **Final Energy**: $(data["method_result"]["vqe_result"].final_energy)\n")
                write(f, "- **Energy Error**: $(data["performance_metrics"]["final_energy_error"])\n")
                write(f, "- **Gradient Variance**: $(data["bp_diagnostics"].gradient_variance)\n")
                write(f, "- **State Fidelity**: $(data["performance_metrics"]["state_fidelity"])\n")
                write(f, "- **Execution Time**: $(round(data["execution_time"], digits=2))s\n")
                write(f, "- **Converged**: $(data["method_result"]["vqe_result"].converged)\n\n")
            end
        end
    end
    
    println("📄 Analysis summary saved to: $summary_path")
end

# ============================================================================
# Exports
# ============================================================================

export plot_energy_convergence, plot_loss_landscape_3d, plot_loss_landscape_contour
export create_comprehensive_visualization, quick_analysis_plot
export create_performance_table, create_latex_table
export compute_loss_landscape_2d, compute_optimization_trajectory_2d
export plot_gradient_diagnostics, plot_method_comparison, generate_all_plots
export save_analysis_summary