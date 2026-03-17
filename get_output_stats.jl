using NCDatasets
# using Images
# using ImageFiltering
# using Statistics
# using DelimitedFiles
# using Printf
# using Trapz
# using Missings

mutable struct XBProcessStats
    num_cpu::Int
    file_dir::String
    path_to_model::String


    domain_dims::Tuple{Int, Int}
    load_current::Bool

    # Constants
    rho::Float64
    g::Float64
    connectivity::Matrix{Bool}
    io_lock::ReentrantLock
    processing_step_size::Int


    function XBHotstartRunner(; model_runname, num_cpu=10, specific_cores=false)
        file_dir = @__DIR__
        path_to_model = joinpath(file_dir, path_to_model)
        
        check_threads(num_cpu)
        domain_size = read_domain_size(path_to_model)
        println(domain_size)
        fds

        # Initialize constants
        rho = 1025.0
        g = 9.81
        io_lock = ReentrantLock()
        processing_step_size = 100   # number of x-slices to hold in RAM at once
        

        new(
            num_cpu, 
            file_dir, 
            path_to_model, 

            domain_size,
            false,

            rho, 
            g, 
            connectivity, 
            io_lock, 
            processing_step_size,
            )
    end
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
                nx = parse(Float64, m_dx.captures[1])
            elseif m_ny !== nothing
                ny = parse(Float64, m_dy.captures[1])
            end
            
            # Stop early if both are found
            if !isnothing(nx) && !isnothing(ny)
                break
            end
        end
    end
    return (nx, ny)
end

function read_all_hotstarts(path_to_models::String)
    dirs = readdir(path_to_models)
    dirs = [i for i in dirs if occursin("hotstart", i)]
    dirs = sort(dirs)
    return dirs
end

function check_threads(target_threads::Int)
    if Threads.nthreads() != target_threads
        # This replaces the current process with a new one with -t flags
        run(`$(Base.julia_cmd()) -t $target_threads  --project=$(Base.active_project()) $(PROGRAM_FILE)`)
        exit()
    end
end

# --- Main Logic ---
function run_simulation!(r::XBHotstartRunner)
    println("Running XBeach in hotstart mode")
    println("  hotstart simulations: $(r.hot_starts)")
    flush(stdout)

    for (i, hot_start) in enumerate(r.hot_starts)
        println("Starting Hotstart: $hot_start")
        flush(stdout)

        if i > 1 # if i is greater than 1, then read cumulative impulse from previous output files
            fn_cip = joinpath(r.path_to_models, "$(r.hot_starts[i-1])", "stat_cumulative_horizontal_impulse.dat")
            r.cumulative_horizontal_impulse = readdlm(fn_cip)

            fn_cip = joinpath(r.path_to_models, "$(r.hot_starts[i-1])", "stat_cumulative_uplift_impulse.dat")
            r.cumulative_uplift_impulse = readdlm(fn_cip)

            if r.load_current
                fn_ccri = joinpath(r.path_to_models, "$(r.hot_starts[i-1])", "stat_cumulative_current_impulse.dat")
                r.cumulative_current_impulse = readdlm(fn_ccri)
            end
        end

        if i<=17
            continue
        elseif i>18
            run_xbeach(r, hot_start)
        end

        start = time_ns()
        # Update paths for current and next
        set_paths!(r, i)
        if i < length(r.hot_starts)
            cp_hotstart_files(r)
            compute_wave_stats!(r, i)
            compute_uplift_forces(r)
            bldgs_remove = get_destroyed_bldgs(r)

            edit_zs(r, bldgs_remove)
            edit_zb(r, bldgs_remove)
            update_bldgs!(r)
            remove_xboutput(r)
        else
            # Final Run
            compute_wave_stats!(r, i)
            compute_uplift_forces(r)
            bldgs_remove = get_destroyed_bldgs_final(r)
            fn_out = joinpath(r.path_to_models, "stat_removed_bldgs.dat")
            write_fortran(fn_out, bldgs_remove)
            remove_xboutput(r)
        end
        elapsed = (time_ns() - start) / 1e9  # elapsed time; converting nano second to seconds
        sec_str = @sprintf("%.3f", elapsed)
        hr_str  = @sprintf("%.3f", elapsed / 3600)

        println("\n\n")
        println("num bldgs start iteration: $(maximum(r.bldgs_original))")
        println("num bldgs end iteration: $(maximum(r.bldgs))")
        println("elapsed time processing results: $sec_str sec ($hr_str hr)")
        println("\n\n")
        flush(stdout)
    end
end

# --- Helper Functions ---
function cp_hotstart_files(r)
    files0 = readdir(r.path_hs0)
    files0 = [i for i in files0 if occursin("hotstart", i)]
    for file in files0
        src = joinpath(r.path_hs0, file)
        dst = joinpath(r.path_hs1, file)
        cp(src, dst, force=true)
    end
end

function set_paths!(r::XBHotstartRunner, i::Int)
    r.path_hs0 = joinpath(r.path_to_models, "$(r.hot_starts[i])")
    if i < length(r.hot_starts)
        r.path_hs1 = joinpath(r.path_to_models, "$(r.hot_starts[i+1])")
    end
end

function run_xbeach(r::XBHotstartRunner, hot_start::String)
    path = joinpath(r.path_to_models, "$(hot_start)")
    cd(path) do
        cmd = if r.specific_cores == false
            `mpiexec -np $(r.num_cpu) xbeach`
        else
            `mpiexec -np $(r.num_cpu) --cpu-set $(r.specific_cores) --bind-to core xbeach`
        end
        run(cmd)
    end
end

function compute_wave_stats!(r::XBHotstartRunner, hs_i::Int)
    fn_nc = joinpath(r.path_hs0, "xboutput.nc")
    Hs = Matrix{Float32}(undef, r.domain_dims...)
    Hmax = Matrix{Float32}(undef, r.domain_dims...)
    impulse = Matrix{Float32}(undef, r.domain_dims...)
    water_elev = Matrix{Float32}(undef, r.domain_dims...)
    max_zs = Matrix{Float32}(undef, r.domain_dims...)
    max_curr = Matrix{Float32}(undef, r.domain_dims...)
    max_curr_impulse = Matrix{Float32}(undef, r.domain_dims...)

    fn_grid = joinpath(r.path_hs1, "z.grd")
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
                    water_elev[y_, x_idx] = compute_water_elev(z)           # compute water elevation (e.g., surge)
                    max_zs[y_, x_idx]     = compute_max_zs(z)               # compute max water elevation

                    if r.load_current
                        ue = @view ue_slab[x_local, y_,:]
                        ve = @view ve_slab[x_local, y_,:]
                        max_curr[y_, x_idx] = compute_current(ue, ve)
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
    replace!(max_zs, NaN => 0.0)             # replace NaN with 0.0 in max_zs
    r.cumulative_horizontal_impulse .+= impulse     # add impulse for iteration to cumulative impulse
    if r.load_current
        r.cumulative_current_impulse .+= max_curr_impulse
    end

    # setting file names.
    fn_hmx = joinpath(r.path_hs0, "stat_Hmax.dat")
    fn_hs  = joinpath(r.path_hs0, "stat_Hs.dat")
    fn_ip  = joinpath(r.path_hs0, "stat_horizontal_impulse.dat")
    fn_cip = joinpath(r.path_hs0, "stat_cumulative_horizontal_impulse.dat")
    fn_wd  = joinpath(r.path_hs0, "stat_water_elev_out.dat")
    fn_mx  = joinpath(r.path_hs0, "stat_max_zs.dat")
    
    # writing out results. 
    write_fortran(fn_hs, Hs)
    write_fortran(fn_hmx, Hmax)
    write_fortran(fn_ip, impulse)
    write_fortran(fn_cip, r.cumulative_horizontal_impulse)
    write_fortran(fn_wd, water_elev)
    write_fortran(fn_mx, max_zs)
    if r.load_current
        replace!(max_curr, NaN => 0.0)
        replace!(max_curr_impulse, NaN => 0.0)
        replace!(r.cumulative_current_impulse, NaN => 0.0)

        fn_cr = joinpath(r.path_hs0, "stat_max_current_vel.dat")
        fn_cri = joinpath(r.path_hs0, "stat_max_current_impulse.dat")
        fn_ccri = joinpath(r.path_hs0, "stat_cumulative_current_impulse.dat")
        
        write_fortran(fn_cr, max_curr)
        write_fortran(fn_cri, max_curr_impulse)
        write_fortran(fn_ccri, r.cumulative_current_impulse)
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
                        r::XBHotstartRunner)
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
    current_mag = sqrt.((ue.^2)+(ve.^2))
    max_curr = maximum(filter(!isnan, skipmissing(current_mag)); init=0)
    return max_curr
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
        r::XBHotstartRunner)
    if all(ismissing, h) return 0.0 end # if data is missing, impulse = 0
    current_mag = sqrt.((ue.^2)+(ve.^2))
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

function read_threshold(path)
    fn = joinpath(path, "threshold.grd")
    data = readdlm(fn)
    return data
end

function read_all_bldgs(path, connectivity)
    fn = joinpath(path, "z.grd")
    data = readdlm(fn)
    bldgs = data.==10.0
    bldgs = label_components(bldgs, connectivity)
    return bldgs
end

function read_elevated_bldgs(path, connectivity)
    fn = joinpath(path, "elevated_bldgs.grd")
    bldg_ffe = readdlm(fn)
    bldgs = bldg_ffe.>0
    elevated_labels = label_components(bldgs, connectivity)
    return bldg_ffe, elevated_labels
end

neighbor_footprint() = Bool[0 1 0; 1 1 1; 0 1 0]

function get_destroyed_bldgs(r::XBHotstartRunner)
    removed_bldgs = falses(size(r.bldgs))
    num_bldgs = maximum(r.bldgs)
    for i = 1:num_bldgs
        mask_temp = (r.bldgs .== i)   # get mask for current building
        dilated_labels = dilate(mask_temp, r.connectivity)  # dilating building to get neighboring cells
        neighbor_mask = (dilated_labels .!= 0) .& (mask_temp .== 0) .& (r.bldgs .== 0)  # get bldg's neighboring cells; dilated area minus building area minus other buildings

        threshold_ = maximum(r.threshold[mask_temp])
                
        # Get indices of neighbors
        idxs = findall(neighbor_mask) 
        max_val = 0.0

        for idx in idxs
            d_ = r.cumulative_horizontal_impulse[idx]
            if d_>max_val
                max_val = d_
            end
        end
        if max_val > threshold_
            removed_bldgs[mask_temp] .= true
        end
    end
    return removed_bldgs
end
function read_resolution(r::XBHotstartRunner)
    # Initialize variables with default values
    dx = nothing
    dy = nothing

    # Open the file and iterate through lines
    fn_params = joinpath(r.path_hs0, "params.txt")
    open(fn_params, "r") do file
        for line in eachline(file)
            # Match: "variable name", optional spaces, "=" or ":", then the "value"
            m_dx = match(r"^dx\s*[=:]\s*([\d\.]+)", line)
            m_dy = match(r"^dy\s*[=:]\s*([\d\.]+)", line)

            if m_dx !== nothing
                dx = parse(Float64, m_dx.captures[1])
            elseif m_dy !== nothing
                dy = parse(Float64, m_dy.captures[1])
            end
            
            # Stop early if both are found
            if !isnothing(dx) && !isnothing(dy)
                break
            end
        end
    end
    return dx, dy
end

function compute_uplift_forces(r::XBHotstartRunner)
    uplift_impulse = fill(Float32(0), r.domain_dims...)
    num_bldgs = maximum(r.elevated_labels)
    
    # read grid elevation data; used to compute depth
    fn_grid = joinpath(r.path_hs1, "z.grd")
    z = readdlm(fn_grid)
    dx, dy = read_resolution(r)

    # read xboutput. lazy loading for now.
    fn_nc = joinpath(r.path_hs0, "xboutput.nc")
    NCDataset(fn_nc) do ds
        t = ds["globaltime"][:]     # time array
        dt = t[2] - t[1]            # time step
        zs_raw = ds["zs"]           # lazy loading zs       
        for i = 1:num_bldgs
            mask_temp = (r.elevated_labels .== i)
            dilated_labels = dilate(mask_temp, r.connectivity)
            perimeter = (dilated_labels .!= 0) .& (mask_temp .== 0) .& (r.bldgs .== 0)
            ffe = r.elevated_bldg_ffe[mask_temp][1]
            floor_area = sum(mask_temp)*dx*dy   # units are m^2; number of cells*dx*dy

            # get indices of perimeter; note that xboutput is (x,y,t); The input grid is (y,x)
            idxs_org = findall(x->x==1, perimeter)                              # indices as (x,y)
            idxs_inv = [CartesianIndex(y,x) for (x,y) in Tuple.(idxs_org)]      # indices as (y,x)

            # read in data only at perimeter
            zs_data = zs_raw[idxs_inv,:]    # water elevation
            z_data = z[idxs_org]            # grid elevation
            depth = zs_data .- z_data       # water detph
            free_board = depth .- ffe       # free board; difference from depth to ffe; rows are perimeter cells, cols are time steps

            # move through each column (time step) and compute mean detph
            fb_perimeter = map(eachcol(free_board)) do col
                filtered = skipmissing(col)
                isempty(filtered) ? 0 : mean(filtered)
            end
            fb_perimeter[fb_perimeter.<0] .= 0

            # computing uplift force; F_u = rho*g*V = rho*g*A*fb
            uplift_force = r.rho * r.g * floor_area * fb_perimeter              # units are N

            # taking maximum uplift force 
            impulse = nantrapz(uplift_force, t)                                 # units are N-s
            impulse = impulse/3600                                              # units are now N-hr
            impulse = impulse/1000                                              # units are now kN-hr
            uplift_impulse[mask_temp.==1] .= impulse
        end
    end

    r.cumulative_uplift_impulse .+= uplift_impulse
    fn_uplift  = joinpath(r.path_hs0, "stat_uplift_impulse.dat")
    fn_cuplift  = joinpath(r.path_hs0, "stat_cumulative_uplift_impulse.dat")

    write_fortran(fn_uplift, uplift_impulse)
    write_fortran(fn_cuplift, r.cumulative_uplift_impulse)

end

function get_destroyed_bldgs_final(r::XBHotstartRunner)
    removed_bldgs = falses(size(r.bldgs_original))
    num_bldgs = maximum(r.bldgs_original)
    for i = 1:num_bldgs
        mask_temp = (r.bldgs_original .== i)   # get mask for current building
        dilated_labels = dilate(mask_temp, r.connectivity)  # dilating building to get neighboring cells
        neighbor_mask = (dilated_labels .!= 0) .& (mask_temp .== 0) .& (r.bldgs_original .== 0)  # get bldg's neighboring cells; dilated area minus building area minus other buildings

        threshold_ = maximum(r.threshold[mask_temp])
                
        # Get indices of neighbors
        idxs = findall(neighbor_mask) 
        max_val = 0.0

        for idx in idxs
            d_ = r.cumulative_horizontal_impulse[idx]
            if d_>max_val
                max_val = d_
            end
        end
        if max_val > threshold_
            removed_bldgs[mask_temp] .= true
        end
    end
    return removed_bldgs
end

function edit_zs(r::XBHotstartRunner, bldgs_remove::BitMatrix)
    fn = joinpath(r.path_hs1, "hotstart_zs000001.dat")
    
    # Read as text
    zs = readdlm(fn)
    zs = Float64.(zs)
    labeled_mask = label_components(bldgs_remove)   # labeling the removed bldgs
    num_bldgs = maximum(labeled_mask)

    for i in 1:num_bldgs
        mask_temp = (labeled_mask .== i)
        dilated_labels = dilate(mask_temp, r.connectivity)
        neighbor_mask = (dilated_labels .!= 0) .& (mask_temp .== 0) .& (labeled_mask .== 0)
        
        vals = zs[neighbor_mask]
        vals = vals[vals .!= 10.0]
        
        if !isempty(vals)
            avg_val = mean(vals)
            zs[mask_temp] .= avg_val
        end
    end
    
    write_fortran(fn, zs)
end


function edit_zb(r::XBHotstartRunner, bldgs_remove::BitMatrix)
    fn_zb_hotstart = joinpath(r.path_hs1, "hotstart_zb000001.dat")
    fn_grid = joinpath(r.path_hs1, "z.grd")
    fn_nobldgs = joinpath(r.path_hs1, "z_nobldgs.grd")
    
    zb_nobldgs = Float64.(readdlm(fn_nobldgs))
    zb_current = Float64.(readdlm(fn_zb_hotstart))
    
    # Julia logical indexing
    zb_current[bldgs_remove] = zb_nobldgs[bldgs_remove]
    
    write_fortran(fn_zb_hotstart, zb_current)
    
    # Save Grid (simple space delimiter)
    open(fn_grid, "w") do io
        writedlm(io, zb_current, ' ') # Default precision is usually fine, or format manually
    end
end

function update_bldgs!(r::XBHotstartRunner)
    connectivity = neighbor_footprint()
    r.bldgs = read_all_bldgs(r.path_hs1, connectivity)
end

remove_xboutput(r::XBHotstartRunner) = rm(joinpath(r.path_hs0, "xboutput.nc"))

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
processor = XBProcessStats(
    path_to_model=joinpath("..", "..", "model-runs", "comp-run9"), 
    num_cpu=2)
process!(processor)



















