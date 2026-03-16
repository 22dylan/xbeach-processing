using NCDatasets: NCDataset
using Statistics: mean, quantile
using DelimitedFiles: readdlm
using Printf: @sprintf, @printf
using Trapz: trapz
using Missings: Missings

mutable struct XBProcessStats
    num_cpu::Int
    file_dir::String
    path_to_model::String
    path_to_output::String

    domain_dims::Tuple{Int, Int}
    load_current::Bool

    # Constants
    rho::Float64
    g::Float64
    io_lock::ReentrantLock
    processing_step_size::Int


    function XBProcessStats(; num_cpu=10)
        file_dir = @__DIR__
        path_to_model, path_to_output = read_paths(file_dir)
        # path_to_model = joinpath(file_dir, path_to_model)
        # path_to_output = joinpath(file_dir, "..", "..", "processed-results", "test-comp-run8")
        
        # check_threads(num_cpu)
        domain_size = read_domain_size(path_to_model)

        # Initialize constants
        rho = 1025.0
        g = 9.81
        io_lock = ReentrantLock()
        processing_step_size = 10   # number of x-slices to hold in RAM at once
        

        new(
            num_cpu, 
            file_dir, 
            path_to_model, 
            path_to_output,

            domain_size,
            false,

            rho, 
            g, 
            io_lock, 
            processing_step_size,
            )
    end
end

function read_paths(file_dir::String)
    # Initialize variables with default values
    path_to_model = nothing
    path_to_output = nothing

    # Open the file and iterate through lines
    fn_params = joinpath(file_dir, "..", "paths.txt")
    open(fn_params, "r") do file
        for line in eachline(file)
            # Match: "variable name", optional spaces, "=", optional spaces, then the "path" (non-whitespace characters)
            pmd = match(r"^path_to_model\s*=\s*(\S+)", line)
            psp = match(r"^path_to_save_plot\s*=\s*(\S+)", line)

            if pmd !== nothing
                path_to_model = String(pmd.captures[1])
            elseif psp !== nothing
                path_to_output = String(psp.captures[1])
            end
            
            # Stop early if both are found
            if !isnothing(path_to_model) && !isnothing(path_to_output)
                break
            end
        end
    end
    path_to_model = joinpath("..", path_to_model)
    path_to_output = joinpath("..", path_to_output)

    return path_to_model, path_to_output
end

function check_threads(target_threads::Int)
    if Threads.nthreads() != target_threads
        # This replaces the current process with a new one with -t flags
        run(`$(Base.julia_cmd()) -t $target_threads  --project=$(Base.active_project()) $(PROGRAM_FILE)`)
        exit()
    end
end

# --- Main Logic ---
function process!(r::XBProcessStats)
    println("Processing results for: ")
    println("  $(r.path_to_model)")
    flush(stdout)

    start = time_ns()           # start time
    compute_output_stats!(r)   # compute wave stats
    elapsed = (time_ns() - start) / 1e9  # elapsed time; converting nano second to seconds
    sec_str = @sprintf("%.3f", elapsed)
    hr_str  = @sprintf("%.3f", elapsed / 3600)

    println("\n\n")
    println("elapsed time processing results: $sec_str sec ($hr_str hr)")
    println("\n\n")
    flush(stdout)

end


function compute_output_stats!(r::XBProcessStats)
    fn_nc = joinpath(r.path_to_model, "xboutput.nc")
    Hs = Matrix{Float32}(undef, r.domain_dims...)
    Hmax = Matrix{Float32}(undef, r.domain_dims...)
    impulse = Matrix{Float32}(undef, r.domain_dims...)
    water_elev = Matrix{Float32}(undef, r.domain_dims...)
    flood_depth = Matrix{Float32}(undef, r.domain_dims...)
    max_zs = Matrix{Float32}(undef, r.domain_dims...)
    max_curr = Matrix{Float32}(undef, r.domain_dims...)
    max_curr_dir = Matrix{Float32}(undef, r.domain_dims...)
    max_curr_impulse = Matrix{Float32}(undef, r.domain_dims...)

    fn_grid = joinpath(r.path_to_model, "z.grd")
    zgr = readdlm(fn_grid)
    zgr = Float32.(zgr)

    NCDataset(fn_nc) do ds
        t = ds["globaltime"][:]     # time array 
        dt = t[2] - t[1]            # time step
        zs_raw = ds["zs"]           # lazy loading zs
        
        if ("uu" in keys(ds)) & ("vv" in keys(ds))
            r.load_current = true
            ue_raw = ds["uu"]
            ve_raw = ds["vv"]
        end

        # Configuration
        nx, ny, nt = size(zs_raw)   # size of zs
        for x_start in 1:r.processing_step_size:nx  # loop through domain in chunks
            x_end = min(x_start + r.processing_step_size - 1, nx)    # getting x_end; 
            zs_slab = zs_raw[x_start:x_end, :, :]   # loading chunk into RAM

            if r.load_current
                ue_slab = ue_raw[x_start:x_end, :, :]
                ve_slab = ve_raw[x_start:x_end, :, :]
            end
            println("processing $(x_start):$(x_end) out of $(nx)")
            flush(stdout)
            Threads.@threads for y_ = 1:ny              # loop through each point in the chunk in parallel; y-first
                for x_local = 1:(x_end - x_start + 1)   # ... and loop through x.
                    x_idx = x_start + x_local - 1       # x-idx used to store data
                    z = @view zs_slab[x_local, y_, :]   # getting a slice of the z-data
                    h = z .- zgr[y_, x_idx]             # water elevation above ground; i.e., depth (includes waves)
                    check_h_for_negative_values!(h)     # replaces negative water depths with 0
                    H = get_wave_heights(z)             # compute wave heights

                    Hs[y_, x_idx]         = compute_Hs(H)                   # compute significant wave height
                    Hmax[y_,x_idx]        = compute_Hmax(H)                 # compute max wave height
                    impulse[y_, x_idx]    = compute_impulse(h, t, dt, r)    # compute impulse
                    water_elev[y_, x_idx] = compute_water_elev(z)           # compute water elevation (e.g., surge, no waves)
                    flood_depth[y_, x_idx] = water_elev[y_, x_idx] - zgr[y_, x_idx] # flood depth above ground
                    max_zs[y_, x_idx]     = compute_max_zs(z)               # compute max water elevation (e.g. surge + waves)

                    if r.load_current
                        ue = @view ue_slab[x_local, y_,:]
                        ve = @view ve_slab[x_local, y_,:]
                        max_curr[y_, x_idx] = compute_current(ue, ve)
                        max_curr_dir[y_, x_idx] = compute_current_dir(ue, ve)
                        max_curr_impulse[y_, x_idx] = compute_current_force(ue, ve, h, t, r)
                    end
                end
            end
        end
    end

    replace!(Hs, NaN => 0.0)             # replace NaN with 0.0 in Hs
    replace!(Hmax, NaN => 0.0)             # replace NaN with 0.0 in Hs
    replace!(impulse, NaN => 0.0)        # replace NaN with 0.0 in impulse
    replace!(water_elev, NaN => 0.0)             # replace NaN with 0.0 in max_zs
    replace!(flood_depth, NaN => 0.0)             # replace NaN with 0.0 in max_zs
    replace!(max_zs, NaN => 0.0)             # replace NaN with 0.0 in max_zs


    # setting file names.
    fn_we  = joinpath(r.path_to_output, "surge_max.dat")
    fn_wd  = joinpath(r.path_to_output, "flood_depth_max.dat")
    
    fn_hmx = joinpath(r.path_to_output, "Hmax.dat")
    fn_hs  = joinpath(r.path_to_output, "Hs_max.dat")
    fn_ip  = joinpath(r.path_to_output, "horizontal_impulse_max.dat")
    fn_mx  = joinpath(r.path_to_output, "zs_max.dat")
    
    # writing out results. 
    write_fortran(fn_we, water_elev)
    write_fortran(fn_wd, flood_depth)
    
    write_fortran(fn_hs, Hs)
    write_fortran(fn_hmx, Hmax)
    write_fortran(fn_ip, impulse)
    write_fortran(fn_mx, max_zs)
    if r.load_current
        replace!(max_curr, NaN => 0.0)
        replace!(max_curr_dir, NaN => 0.0)
        replace!(max_curr_impulse, NaN => 0.0)

        fn_cr = joinpath(r.path_to_output, "velocity_magnitude_max.dat")
        fn_cd = joinpath(r.path_to_output, "velocity_direction_max.dat")
        fn_cri = joinpath(r.path_to_output, "current_impulse_max.dat")
        
        write_fortran(fn_cr, max_curr)
        write_fortran(fn_cd, max_curr_dir)
        write_fortran(fn_cri, max_curr_impulse)
    end
end

function check_h_for_negative_values!(h::Vector{<:Union{Missing, Float32}})
    for i in eachindex(h)
        if !ismissing(h[i]) && h[i] < 0
            h[i] = 0
        end
    end
end
compute_max_zs(z::SubArray{<:Union{Missing, Float32}}) = maximum(filter(!isnan, skipmissing(z)); init=0)
function compute_water_elev(z::SubArray{<:Union{Missing, Float32}})
    total = 0.0f0
    count = 0
    
    for x in z
        if !ismissing(x) && !isnan(x)
            total += x
            count += 1
        end
    end
    return count == 0 ? NaN32 : total / count
end
function compute_impulse(
                        # z::SubArray{<:Union{Missing, Float32}}, 
                        h::Vector{<:Union{Missing, Float32}}, 
                        t::Vector{Float32}, 
                        dt::Float32, 
                        r::XBProcessStats)
    if all(ismissing, h) return 0.0 end # if data is missing, impulse = 0
    h_rm = running_mean(h, dt)     # running mean for water depth    
    eta = h - h_rm                 # eta
    # calculate wave force
    fw = ((r.rho*r.g)/2) .* abs.((2*h_rm.*eta) + (abs.(eta).^2))  # units are N/m
    # integrate across time
    I = nantrapz(fw, t)    # units are (N/m)-s
    I = I/3600                              # units are now (N-hr)/m
    I = I/1000                              # units are now (kN-hr)/m
    return ismissing(I) ? 0.0 : I
end

""" compute maximum current velocity """
function compute_current(ue::SubArray{<:Union{Missing, Float32}}, ve::SubArray{<:Union{Missing, Float32}})
    current_mag = hypot.(ue, ve)
    max_curr = maximum(filter(!isnan, skipmissing(current_mag)); init=0)
    return max_curr
end

function compute_current_dir(ue::SubArray{<:Union{Missing, Float32}}, ve::SubArray{<:Union{Missing, Float32}})
    # Calculate magnitudes. `hypot.` is idiomatic Julia and safer than sqrt(u^2 + v^2) 
    # as it prevents overflow/underflow.
    mag = hypot.(ue, ve)
    
    # Check if the entire array consists of NaNs
    if all(isnan, mag)
        return NaN
    end
    
    # Find the index of the max magnitude. 
    # Since Julia doesn't have a direct `nanargmax` in Base, we temporarily 
    # replace NaNs with -Inf so argmax ignores them.
    max_idx = argmax(replace(mag, NaN => -Inf))
    
    # Extract the components at the maximum magnitude
    u_max = ue[max_idx]
    v_max = ve[max_idx]
    
    # Calculate the angle. atan(y, x) is Julia's equivalent to np.arctan2
    angle_deg = rad2deg(atan(v_max, u_max))
    
    # Convert negative angles to the [0, 360) range
    return angle_deg < 0 ? angle_deg + 360.0 : angle_deg

end

"""
From Don's code:
c  calculate water velocity forces:  These are depth averaged currents
c  the water depth is z1  and these forces are per unit width of wall 
c  normal to the current 
          rhowater = 1.94  !  slugs/ft**3  
c  The assumed C_d coefficient of Drag is 1.0 for the wall 
          waterforce(j) =  z1(j) * vspeed(j) **2. * rhowater * 0.5  

Dylan is using same approach, but integrating with time to get impulse (kN-hr/m)
"""
function compute_current_force(
        ue::SubArray{<:Union{Missing, Float32}},
        ve::SubArray{<:Union{Missing, Float32}}, 
        h::Vector{<:Union{Missing, Float32}},
        t::Vector{Float32}, 
        r::XBProcessStats)
    if all(ismissing, h) return 0.0 end # if data is missing, impulse = 0
    
    current_mag = hypot.(ue, ve)
    force = h .* (current_mag.^2) .* (r.rho*0.5)     # units: N/m
    I = nantrapz(force, t)                          # units: (N-s)/m
    I = I/3600                                      # units: (N-hr)/m
    I = I/1000                                      # units: (kN-hr)/m
    return I
end

function nantrapz(data, t)
    mask = .!ismissing.(data)
    if data[mask] isa Vector{Missing}
        clean_t = t
        clean_data = data
    else
        clean_t = t[mask]
        clean_data = collect(Missings.replace(data[mask], 0.0)) # Ensure type is Vector{Float64}
    end
    area = trapz(clean_t, clean_data)
    return area
end

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

function running_mean(z, dt; window_size_sec=120)
    window_size = window_size_sec/dt
    window_size = convert(Int, window_size)
    
    run_mean = moving_average(z, window_size)
    n = length(z) - length(run_mean)
    prepend!(run_mean, fill(run_mean[1], n))
    return run_mean
end

function get_wave_heights(z::SubArray{<:Union{Missing, Float32}}; detrend=true)
    if detrend z = z .- mean(skipmissing(z)) end            # detrend data

    signs = sign.(z)
    zero_crossings, up_crossings = get_crossings(z, signs)
    if length(up_crossings) < 2 return Float32[0.0] end
    heights = Float32[]
    for i in 1:length(up_crossings)-1
        segment = z[up_crossings[i]:up_crossings[i+1]]
        segment = filter(!ismissing, segment)
        push!(heights, maximum(segment) - minimum(segment))
    end
    return heights
end

function get_crossings(z::Vector{<:Union{Missing, Float32}}, signs::Vector{<:Union{Missing, Float32}})
    d = diff(signs)
    zero_crossings = Int[]
    up_crossings = Int[]
    cnt = 1
    for d_ in d
        if ismissing(d_)
            cnt += 1
            continue
        end
        if d_ != 0
            push!(zero_crossings, cnt)
            if d_ == 2
                push!(up_crossings,cnt)
            end
        end
        cnt += 1
    end
    return zero_crossings, up_crossings
end

compute_Hmax(z::Vector{<:Union{Missing,Float32}}) = maximum(filter(!isnan, skipmissing(z)); init=0)
function compute_Hs(heights::Vector{<:Union{Missing,Float32}})
    if isempty(heights) return 0.0 end
    q = quantile(heights, 2/3)
    top_third = heights[heights .>= q]
    return isempty(top_third) ? 0.0 : mean(top_third)
end

function read_domain_size(path_to_model)
    # Initialize variables with default values
    nx = nothing
    ny = nothing

    # Open the file and iterate through lines
    fn_params = joinpath(path_to_model, "params.txt")
    open(fn_params, "r") do file
        for line in eachline(file)
            # Match: "variable name", optional spaces, "=" or ":", then the "value"
            m_nx = match(r"^nx\s*[=:]\s*([\d\.]+)", line)
            m_ny = match(r"^ny\s*[=:]\s*([\d\.]+)", line)

            if m_nx !== nothing
                nx = parse(Int64, m_nx.captures[1])
            elseif m_ny !== nothing
                ny = parse(Int64, m_ny.captures[1])
            end
            
            # Stop early if both are found
            if !isnothing(nx) && !isnothing(ny)
                break
            end
        end
    end
    return (nx+1, ny+1)
end

# --- Fortran Formatting in Julia ---
function write_fortran(fn, data)
    open(fn, "w") do f
        rows, cols = size(data)
        for i in 1:rows
            for j in 1:cols
                @printf(f, "%16.8E", data[i, j])
            end
            write(f, "\n")
        end
    end
end

# --- Main Execution ---
processor = XBProcessStats(num_cpu=1)
process!(processor)



















