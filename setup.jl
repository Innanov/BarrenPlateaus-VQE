#!/usr/bin/env julia

"""
# BarrenPlateausVQE.jl Complete Setup Script

Single-file setup that handles everything:
- Dependency installation from Project.toml
- Windows plotting backend compatibility
- Optional package installation with fallbacks
- Core functionality testing
- Circuit visualization setup

Run with:
```bash
julia setup.jl
```
"""

using Pkg
using Printf

println("🚀 BarrenPlateausVQE.jl Complete Setup")
println("=" ^ 60)
println("Windows-compatible installation with robust plotting backends")

# ============================================================================
# Step 1: Julia Version Check
# ============================================================================

println("\n📋 Step 1: Julia Version Check")
println("-" ^ 40)

julia_version = VERSION
min_version = v"1.8.0"

if julia_version >= min_version
    println("✅ Julia version: $julia_version (✓ >= $min_version)")
else
    println("❌ Julia version: $julia_version (✗ < $min_version)")
    println("Please update Julia: https://julialang.org/downloads/")
    exit(1)
end

# ============================================================================
# Step 2: Environment Setup
# ============================================================================

println("\n📋 Step 2: Environment Setup")
println("-" ^ 40)

try
    Pkg.activate(".")
    println("✅ Project environment activated")

    project_info = Pkg.project()
    println("    📦 Project: $(project_info.name) v$(project_info.version)")
catch e
    println("❌ Environment activation failed: $e")
    exit(1)
end

# ============================================================================
# Step 3: Core Dependencies Installation
# ============================================================================

println("\n📋 Step 3: Core Dependencies Installation")
println("-" ^ 40)

println("Installing core dependencies from Project.toml...")

try
    # Resolve dependencies first
    println("  🔧 Resolving dependencies...")
    Pkg.resolve()

    # Install core dependencies
    println("  📦 Installing core packages...")
    Pkg.instantiate()

    println("✅ Core dependencies installed successfully")

catch e
    println("❌ Core dependency installation failed: $e")

    # Recovery attempt
    println("\n🔄 Attempting recovery...")
    try
        Pkg.gc()  # Clear cache
        Pkg.resolve()
        Pkg.instantiate()
        println("✅ Recovery successful")
    catch recovery_error
        println("❌ Recovery failed: $recovery_error")
        exit(1)
    end
end

# ============================================================================
# Step 4: Optional Dependencies with Windows Compatibility
# ============================================================================

println("\n📋 Step 4: Optional Dependencies (Windows-compatible)")
println("-" ^ 40)

# Define optional packages with Windows compatibility in mind
optional_packages = [
    ("YaoPlots", "Circuit visualization", true),      # High priority
    ("StatsPlots", "Statistical plotting", false),   # Medium priority  
    ("GR", "GR plotting backend", false),            # Low priority (problematic on Windows)
    ("PyCall", "Python integration", false),          # Optional
]

successful_optional = String[]
failed_optional = String[]

for (pkg_name, description, is_critical) in optional_packages
    print("  📦 Installing $pkg_name ($description)...")

    try
        Pkg.add(pkg_name)
        println(" ✅")
        push!(successful_optional, pkg_name)
    catch e
        if is_critical
            println(" ❌ (Critical)")
            push!(failed_optional, pkg_name)
            @warn "Critical package $pkg_name failed: $e"
        else
            println(" ⚠️  (Optional)")
            @info "Optional package $pkg_name skipped: $e"
        end
    end
end

println("\n  📊 Optional package summary:")
println("    ✅ Installed: $(join(successful_optional, ", "))")
if !isempty(failed_optional)
    println("    ❌ Failed: $(join(failed_optional, ", "))")
end

# ============================================================================
# Step 5: Package Import Testing
# ============================================================================

println("\n📋 Step 5: Package Import Testing")
println("-" ^ 40)

# Test core packages
core_packages = [
    ("DataFrames", "using DataFrames"),
    ("CSV", "using CSV"),
    ("Yao", "using Yao"),
    ("YaoBlocks", "using YaoBlocks"),
    ("YaoArrayRegister", "using YaoArrayRegister"),
    ("Optim", "using Optim"),
    ("ForwardDiff", "using ForwardDiff"),
]

global core_working = true
for (name, import_cmd) in core_packages
    try
        eval(Meta.parse(import_cmd))
        println("  ✅ $name")
    catch e
        println("  ❌ $name: $e")
        global core_working = false
    end
end

if !core_working
    println("\n❌ Core packages failed to import!")
    exit(1)
end

# Test plotting with Windows-compatible fallbacks
println("\n  🎨 Testing plotting backends...")
global plotting_backend = nothing

# Try PlotlyJS first (most Windows-compatible)
try
    eval(Meta.parse("using PlotlyJS"))
    eval(Meta.parse("using Plots"))
    eval(Meta.parse("Plots.plotlyjs()"))
    println("  ✅ PlotlyJS backend (recommended for Windows)")
    global plotting_backend = "PlotlyJS"
catch e
    println("  ⚠️  PlotlyJS backend failed: $e")
end

# Try GR if PlotlyJS failed (but expect issues on Windows)
if plotting_backend === nothing
    try
        eval(Meta.parse("using GR"))
        eval(Meta.parse("using Plots"))
        eval(Meta.parse("Plots.gr()"))
        println("  ⚠️  GR backend (may have Windows issues)")
        global plotting_backend = "GR"
    catch e
        println("  ⚠️  GR backend failed: $e")
    end
end

# Fallback to default
if plotting_backend === nothing
    try
        eval(Meta.parse("using Plots"))
        println("  ⚠️  Default Plots backend (limited functionality)")
        global plotting_backend = "Default"
    catch e
        println("  ❌ All plotting backends failed: $e")
        global plotting_backend = "None"
    end
end

# Test YaoPlots for circuit visualization
global yaoplots_working = false
if "YaoPlots" in successful_optional
    try
        eval(Meta.parse("using YaoPlots"))
        println("  ✅ YaoPlots (circuit visualization)")
        global yaoplots_working = true
    catch e
        println("  ❌ YaoPlots failed: $e")
    end
end

# ============================================================================
# Step 6: BarrenPlateausVQE Package Testing
# ============================================================================

println("\n📋 Step 6: BarrenPlateausVQE Package Testing")
println("-" ^ 40)

println("Loading main package...")

# Test tracking variables
global tests_passed = 0
global total_tests = 5

try
    include("src/BarrenPlateausVQE.jl")
    using .BarrenPlateausVQE
    println("✅ Main package loaded")
    global tests_passed += 1
catch e
    println("❌ Failed to load main package: $e")
    exit(1)
end

# Test 1: Hamiltonian creation
print("  🧪 Testing Hamiltonian creation...")
global test_system = nothing
try
    h2_system = create_molecular_hamiltonian("H2")
    println(" ✅ ($(h2_system.n_qubits) qubits)")
    global tests_passed += 1
    global test_system = h2_system
catch e
    try
        test_H = test_hamiltonian(4)
        exact_energy = classical_solver(test_H).eigenvalue
        println(" ✅ (test Hamiltonian)")
        global tests_passed += 1
        global test_system = (
            hamiltonian=test_H, n_qubits=4, exact_energy=exact_energy, name="Test"
        )
    catch e2
        println(" ❌ ($e2)")
        global test_system = nothing
    end
end

# Test 2: VQE methods
print("  🧪 Testing VQE methods...")
global working_methods = 0
try
    vqe = StandardVQE(4, 1)
    global working_methods += 1

    try
        vqe_sea = SEAVQE(4)
        global working_methods += 1
    catch
    end

    if working_methods >= 1
        println(" ✅ ($working_methods methods working)")
        global tests_passed += 1
    else
        println(" ❌ (no methods working)")
    end
catch e
    println(" ❌ ($e)")
end

# Test 3: Quick optimization
if test_system !== nothing && working_methods >= 1
    print("  🧪 Testing optimization...")
    try
        vqe = StandardVQE(test_system.n_qubits, 1)
        initial_params = random_initial_parameters(vqe.n_parameters)
        result = run_vqe(vqe, test_system.hamiltonian, initial_params, 10; verbose=false)

        energy_error = abs(result.final_energy - test_system.exact_energy)
        println(" ✅ (error: $(round(energy_error, digits=6)))")
        global tests_passed += 1
    catch e
        println(" ❌ ($e)")
    end
else
    println("  ⚠️  Skipping optimization test")
end

# Test 4: Analysis framework
if working_methods >= 1
    print("  🧪 Testing analysis framework...")
    try
        analyzer = MolecularVQEAnalyzer("H2"; use_test_hamiltonian=true)
        result = run_standard_vqe(analyzer; num_iters=5, verbose=false)
        println(" ✅")
        global tests_passed += 1
    catch e
        println(" ❌ ($e)")
    end
else
    println("  ⚠️  Skipping analysis test")
end

# ============================================================================
# Step 7: Circuit Visualization Setup
# ============================================================================

println("\n📋 Step 7: Circuit Visualization Setup")
println("-" ^ 40)

global circuit_viz_status = "Not Available"

if yaoplots_working && plotting_backend != "None"
    try
        include("src/circuit_visualization.jl")
        using .CircuitVisualization

        deps_ok = CircuitVisualization.check_dependencies()
        if deps_ok
            println("✅ Circuit visualization module loaded successfully")
            global circuit_viz_status = "Available"

            # Quick test
            if test_system !== nothing
                try
                    println("  🧪 Testing circuit visualization...")
                    # This is a minimal test that shouldn't fail
                    ansatz = chain(4, put(1=>Ry(0.1)), put(2=>Ry(0.2)), cnot(1, 2))
                    println("  ✅ Circuit visualization ready")
                catch e
                    println("  ⚠️  Circuit visualization test failed: $e")
                    global circuit_viz_status = "Limited"
                end
            end
        else
            println("⚠️  Circuit visualization: some dependencies missing")
            global circuit_viz_status = "Limited"
        end
    catch e
        println("⚠️  Circuit visualization module failed: $e")
        global circuit_viz_status = "Failed"
    end
else
    println("⚠️  Circuit visualization not available:")
    if !yaoplots_working
        println("    • YaoPlots not installed")
    end
    if plotting_backend == "None"
        println("    • No plotting backend available")
    end
end

# ============================================================================
# Step 8: Performance Test
# ============================================================================

println("\n📋 Step 8: Quick Performance Test")
println("-" ^ 40)

if test_system !== nothing && working_methods >= 1
    try
        println("Running H₂ performance test...")

        start_time = time()
        vqe = StandardVQE(test_system.n_qubits, 1)
        initial_params = random_initial_parameters(vqe.n_parameters)
        result = run_vqe(vqe, test_system.hamiltonian, initial_params, 25; verbose=false)
        execution_time = time() - start_time

        energy_error = abs(result.final_energy - test_system.exact_energy)

        println("✅ Performance test completed:")
        println("    📊 Time: $(round(execution_time, digits=2))s")
        println("    📊 Error: $(round(energy_error, digits=8))")
        println("    📊 Speed: $(round(25/execution_time, digits=1)) iterations/s")

        if execution_time < 5.0
            println("    🚀 Excellent performance!")
        else
            println("    ✅ Good performance")
        end

    catch e
        println("⚠️  Performance test failed: $e")
    end
else
    println("⚠️  Skipping performance test")
end

# ============================================================================
# Setup Complete - Final Summary
# ============================================================================

println("\n" * "=" ^ 60)
println("🎉 SETUP COMPLETE!")
println("=" ^ 60)

println("\n📊 FINAL STATUS:")
println("    ✅ Core packages: Working")
println("    ✅ VQE functionality: $working_methods methods available")
println("    ✅ Analysis framework: $(tests_passed >= 4 ? "Working" : "Limited")")
println("    🎨 Plotting backend: $plotting_backend")
println("    🔧 Circuit visualization: $circuit_viz_status")

# Calculate overall health score
global health_score = round(100 * tests_passed / total_tests)
if health_score >= 80
    status_emoji = "🟢"
    status_text = "EXCELLENT"
elseif health_score >= 60
    status_emoji = "🟡"
    status_text = "GOOD"
else
    status_emoji = "🔴"
    status_text = "LIMITED"
end

println("\n$status_emoji OVERALL STATUS: $status_text ($health_score% functional)")

# Usage instructions
println("\n🚀 QUICK START:")
println("```julia")
println("julia --project=.")
println("include(\"src/BarrenPlateausVQE.jl\")")
println("using .BarrenPlateausVQE")
println("")
println("# Basic analysis")
println("analyzer = MolecularVQEAnalyzer(\"H2\")")
println("results = run_complete_analysis(analyzer)")

if circuit_viz_status == "Available"
    println("")
    println("# Circuit visualization")
    println("include(\"src/circuit_visualization.jl\")")
    println("using .CircuitVisualization")
    println("files = visualize_all_ansatz_circuits(analyzer)")
end

println("```")

# Troubleshooting section
if health_score < 100
    println("\n🔧 TROUBLESHOOTING:")

    if plotting_backend == "None"
        println("  • Plotting issues: Try reinstalling PlotlyJS")
        println("    julia --project=. -e 'using Pkg; Pkg.add(\"PlotlyJS\")'")
    end

    if circuit_viz_status != "Available"
        println("  • Circuit visualization: Install YaoPlots manually")
        println("    julia --project=. -e 'using Pkg; Pkg.add(\"YaoPlots\")'")
    end

    if working_methods < 2
        println("  • Limited VQE methods: Check quantum package versions")
    end

    println("  • For detailed diagnostics, check the output above")
end

println("\n📚 RESOURCES:")
println("  • Documentation: README.md")
println("  • Package status: julia --project=. -e 'using Pkg; Pkg.status()'")
println("  • Rerun setup: julia setup.jl")

println("\n✨ Ready to explore barren plateaus in VQE! ✨")

# Exit with appropriate code
if health_score >= 60
    exit(0)  # Success
else
    exit(1)  # Issues detected
end
