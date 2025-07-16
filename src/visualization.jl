"""
# Visualization Module (`visualization.jl`)

High-performance visualization tools for molecular VQE barren plateau analysis.

Key Features:
- Energy convergence plots
- Gradient diagnostics visualization
- Performance comparison tables
- Loss landscape analysis (optional)
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
    catch
        Plots.gr()  # Use GR backend as fallback
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
    safe_bar(args...; kwargs...)

Safely create a bar plot.
"""
function safe_bar(args...; kwargs...)
    if !check_plotting_available()
        return create_text_placeholder("Plots.jl not available")
    end
    
    try
        return PLOTS_MODULE.bar(args...; kwargs...)
    catch e
        @warn "Bar plot creation failed: $e"
        return create_text_placeholder("Bar plot creation failed")
    end
end

"""
    safe_scatter(args...; kwargs...)

Safely create a scatter plot.
"""
function safe_scatter(args...; kwargs...)
    if !check_plotting_available()
        return create_text_placeholder("Plots.jl not available")
    end
    
    try
        return PLOTS_MODULE.scatter(args...; kwargs...)
    catch e
        @warn "Scatter plot creation failed: $e"
        return create_text_placeholder("Scatter plot creation failed")
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
# Visualization Functions
# ============================================================================

"""
    plot_energy_convergence(analyzer::MolecularVQEAnalyzer; 
                           save_path::Union{String, Nothing}=nothing,
                           show_exact::Bool=true,
                           title_override::Union{String, Nothing}=nothing)

Plot energy convergence for all VQE methods.
"""
function plot_energy_convergence(analyzer::MolecularVQEAnalyzer; 
                                save_path::Union{String, Nothing}=nothing,
                                show_exact::Bool=true,
                                title_override::Union{String, Nothing}=nothing)
    
    if isempty(analyzer.results)
        @warn "No results to plot. Run analysis first."
        return create_placeholder_plot("No Results Available", "Run analysis first")
    end
    
    if !check_plotting_available()
        return create_text_placeholder("Plotting not available")
    end
    
    # Create plot
    p = safe_plot(size=(800, 600), dpi=300)
    
    # Plot each method
    colors = [:blue, :red, :green, :purple, :orange, :brown]
    color_idx = 1
    
    for (method_name, data) in analyzer.results
        if get(data["method_result"], "fallback", false)
            continue  # Skip fallback results
        end
        
        energies = data["method_result"]["vqe_result"].energy_history
        iterations = 1:length(energies)
        
        try
            PLOTS_MODULE.plot!(p, iterations, energies, 
                              label=method_name,
                              linewidth=2,
                              color=colors[color_idx],
                              alpha=0.8)
        catch e
            @warn "Failed to add line for $method_name: $e"
        end
        
        color_idx = (color_idx % length(colors)) + 1
    end
    
    # Add exact ground state line
    if show_exact
        try
            PLOTS_MODULE.hline!(p, [analyzer.exact_energy], 
                               label="Exact Ground State",
                               linestyle=:dash,
                               linewidth=2,
                               color=:black,
                               alpha=0.7)
        catch e
            @warn "Failed to add exact energy line: $e"
        end
    end
    
    # Formatting
    try
        PLOTS_MODULE.xlabel!(p, "Iterations")
        PLOTS_MODULE.ylabel!(p, "Energy")
        
        if title_override !== nothing
            PLOTS_MODULE.title!(p, title_override)
        else
            PLOTS_MODULE.title!(p, "Energy Convergence: $(analyzer.molecular_system.name) ($(analyzer.molecular_system.geometry_type))")
        end
    catch e
        @warn "Failed to format plot: $e"
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

"""
    plot_gradient_diagnostics(analyzer::MolecularVQEAnalyzer;
                             save_path::Union{String, Nothing}=nothing)

Plot comprehensive gradient diagnostics.
"""
function plot_gradient_diagnostics(analyzer::MolecularVQEAnalyzer;
                                  save_path::Union{String, Nothing}=nothing)
    
    if isempty(analyzer.results)
        @warn "No results to plot. Run analysis first."
        return create_placeholder_plot("No Results Available", "Run analysis first")
    end
    
    if !check_plotting_available()
        return create_text_placeholder("Gradient diagnostics plotting not available")
    end
    
    # Extract data
    methods = String[]
    gradient_variances = Float64[]
    gradient_norms = Float64[]
    energy_errors = Float64[]
    fidelities = Float64[]
    
    for (method_name, data) in analyzer.results
        if get(data["method_result"], "fallback", false)
            continue  # Skip fallback results
        end
        
        push!(methods, method_name)
        push!(gradient_variances, data["bp_diagnostics"].gradient_variance)
        push!(gradient_norms, data["bp_diagnostics"].gradient_norm_mean)
        push!(energy_errors, data["performance_metrics"]["final_energy_error"])
        push!(fidelities, data["performance_metrics"]["state_fidelity"])
    end
    
    if isempty(methods)
        @warn "No valid results to plot"
        return create_placeholder_plot("No Valid Results", "All methods failed or returned fallback results")
    end
    
    try
        # Create subplots
        p1 = safe_bar(methods, gradient_variances, 
                     title="Gradient Variance by Method",
                     ylabel="Gradient Variance",
                     yscale=:log10,
                     xrotation=45,
                     color=:steelblue,
                     alpha=0.7)
        
        p2 = safe_scatter(gradient_norms, energy_errors,
                         series_annotations=methods,
                         title="Gradient Norm vs Energy Error",
                         xlabel="Gradient Norm Mean",
                         ylabel="Final Energy Error",
                         yscale=:log10,
                         color=:red,
                         alpha=0.7,
                         markersize=6)
        
        p3 = safe_scatter(gradient_variances, fidelities,
                         series_annotations=methods,
                         title="Gradient Variance vs State Fidelity",
                         xlabel="Gradient Variance",
                         ylabel="State Fidelity",
                         xscale=:log10,
                         color=:green,
                         alpha=0.7,
                         markersize=6)
        
        # Normalized performance comparison
        if length(methods) > 1
            norm_variances = gradient_variances ./ maximum(gradient_variances)
            norm_errors = energy_errors ./ maximum(energy_errors)
            norm_fidelities = (1.0 .- fidelities)  # Invert so lower is better
            
            x_pos = 1:length(methods)
            p4 = PLOTS_MODULE.groupedbar([norm_variances norm_errors norm_fidelities],
                                        bar_position=:grouped,
                                        title="Normalized Performance (lower is better)",
                                        ylabel="Normalized Metric",
                                        label=["Grad Variance" "Energy Error" "1 - Fidelity"],
                                        xticks=(x_pos, methods),
                                        xrotation=45,
                                        alpha=0.7)
        else
            p4 = safe_plot(title="Need multiple methods for comparison")
        end
        
        # Combine plots
        combined_plot = PLOTS_MODULE.plot(p1, p2, p3, p4, 
                                         layout=(2, 2), 
                                         size=(1200, 900),
                                         plot_title="Gradient Diagnostics: $(analyzer.molecular_system.name)")
        
        # Save if requested
        if save_path !== nothing
            try
                PLOTS_MODULE.savefig(combined_plot, save_path)
                println("📊 Gradient diagnostics plot saved to: $save_path")
            catch e
                @warn "Failed to save gradient diagnostics plot: $e"
            end
        end
        
        return combined_plot
        
    catch e
        @warn "Failed to create gradient diagnostics plot: $e"
        return create_text_placeholder("Gradient diagnostics creation failed")
    end
end

"""
    quick_analysis_plot(analyzer::MolecularVQEAnalyzer)

Generate a quick summary plot for immediate analysis.
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
                         size=(600, 400))
            
            PLOTS_MODULE.annotate!(p, 5, 8, PLOTS_MODULE.text("System: $(analyzer.molecular_system.name)", :center, 14))
            PLOTS_MODULE.annotate!(p, 5, 7, PLOTS_MODULE.text("Qubits: $(analyzer.n_qubits)", :center, 12))
            PLOTS_MODULE.annotate!(p, 5, 6, PLOTS_MODULE.text("Layers: $(analyzer.n_layers)", :center, 12))
            PLOTS_MODULE.annotate!(p, 5, 4, PLOTS_MODULE.text("Ready for Analysis!", :center, 16, :green))
            PLOTS_MODULE.annotate!(p, 5, 2, PLOTS_MODULE.text("Run: results = run_complete_analysis(analyzer)", :center, 10, :gray))
            
            return p
        catch e
            return create_text_placeholder("Ready: $(analyzer.molecular_system.name), $(analyzer.n_qubits) qubits")
        end
    end
    
    # Create combined plot with results
    if !check_plotting_available()
        return create_text_placeholder("Analysis complete - plotting not available")
    end
    
    try
        p1 = plot_energy_convergence(analyzer; show_exact=true)
        
        # Quick bar chart of final energies
        methods = String[]
        final_energies = Float64[]
        
        for (method_name, data) in analyzer.results
            if !get(data["method_result"], "fallback", false)
                push!(methods, method_name)
                push!(final_energies, data["method_result"]["vqe_result"].final_energy)
            end
        end
        
        if !isempty(methods)
            p2 = safe_bar(methods, final_energies,
                         title="Final Energies",
                         ylabel="Energy",
                         xrotation=45,
                         color=:orange,
                         alpha=0.7)
            
            # Add exact energy line
            try
                PLOTS_MODULE.hline!(p2, [analyzer.exact_energy], 
                                   label="Exact",
                                   linestyle=:dash,
                                   color=:black)
            catch
                # Continue without exact line if it fails
            end
            
            combined = PLOTS_MODULE.plot(p1, p2, layout=(2, 1), size=(800, 800))
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
# Performance Tables (No plotting dependencies)
# ============================================================================

"""
    create_performance_table(analyzer::MolecularVQEAnalyzer; 
                            save_csv::Union{String, Nothing}=nothing,
                            save_latex::Union{String, Nothing}=nothing)

Create performance comparison table.
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
    
    # Print formatted table
    header = Printf.sprintf("%-20s %-12s %-12s %-15s %-12s %-10s", 
                           "Method", "Energy Error", "Fidelity", "Grad Variance", "Grad Norm", "Time (s)")
    println(header)
    println("-"^length(header))
    
    for i in 1:nrow(df)
        row = Printf.sprintf("%-20s %-12.2e %-12.3f %-15.2e %-12.2e %-10.1f",
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

Create LaTeX formatted table.
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
        fidelity = Printf.sprintf("%.3f", df.State_Fidelity[i])
        grad_var = format_scientific_latex(df.Gradient_Variance[i])
        time_s = Printf.sprintf("%.1f", df.Execution_Time[i])
        
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

Format a number for LaTeX scientific notation.
"""
function format_scientific_latex(value::Float64; digits::Int=2)
    if value == 0.0
        return "0"
    end
    
    exponent = floor(Int, log10(abs(value)))
    mantissa = value / (10.0^exponent)
    
    if abs(exponent) <= 3 && abs(value) >= 0.001
        # Use regular decimal notation
        format_str = "%.$(digits)f"
        return Printf.sprintf(format_str, value)
    else
        # Use scientific notation
        mantissa_format = "%.$(digits-1)f"
        mantissa_str = Printf.sprintf(mantissa_format, mantissa)
        return "$(mantissa_str) \\times 10^{$(exponent)}"
    end
end

# ============================================================================
# Stub Functions for Non-Critical Features
# ============================================================================

"""
    plot_method_comparison(analyzer::MolecularVQEAnalyzer; kwargs...)

Plot method comparison (stub if plotting unavailable).
"""
function plot_method_comparison(analyzer::MolecularVQEAnalyzer; kwargs...)
    if !check_plotting_available()
        return create_text_placeholder("Method comparison plotting not available")
    end
    
    # Implementation would go here...
    return create_text_placeholder("Method comparison not implemented yet")
end

"""
    generate_all_plots(analyzer::MolecularVQEAnalyzer; kwargs...)

Generate all plots (stub if plotting unavailable).
"""
function generate_all_plots(analyzer::MolecularVQEAnalyzer; kwargs...)
    if !check_plotting_available()
        @warn "Plotting not available. Install Plots.jl to enable visualization."
        return String[]
    end
    
    generated_files = String[]
    
    try
        # Generate what we can
        println("🎨 Generating available visualizations...")
        
        # Create performance table (always works)
        output_dir = get(kwargs, :output_dir, "./plots")
        mkpath(output_dir)
        
        prefix = "$(analyzer.molecular_system.name)_$(analyzer.molecular_system.geometry_type)"
        csv_path = joinpath(output_dir, "$(prefix)_performance_table.csv")
        
        create_performance_table(analyzer; save_csv=csv_path)
        push!(generated_files, csv_path)
        
        println("✅ Basic visualization generation complete!")
        
    catch e
        @warn "Failed to generate visualizations: $e"
    end
    
    return generated_files
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

export plot_energy_convergence, plot_gradient_diagnostics, create_performance_table
export plot_method_comparison, generate_all_plots, quick_analysis_plot
export save_analysis_summary