module Common

export print_banner

const banner = """
 ███████████                           █████                      █████                 █████   
░░███░░░░░░█                          ░░███                      ░░███                 ░░███    
 ░███   █ ░   ██████  █████████████   ███████    ██████   ██████  ░███████    ██████   ███████  
 ░███████    ███░░███░░███░░███░░███ ░░░███░    ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░   
 ░███░░░█   ░███████  ░███ ░███ ░███   ░███    ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███    
 ░███  ░    ░███░░░   ░███ ░███ ░███   ░███ ███░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
 █████      ░░██████  █████░███ █████  ░░█████ ░░██████ ░░██████  ████ █████░░████████  ░░█████ 
░░░░░        ░░░░░░  ░░░░░ ░░░ ░░░░░    ░░░░░   ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░                                                                                                                                                                                                                                                                                           
"""

print_banner = println(banner)

struct GPURule{F}
    pattern::Regex
    value::F
end

function gpu_regex(patterns)
    patterns = replace.(patterns, r"([\\.^$|?*+()[\]{}])" => s"\\\1")
    Regex("^" * join("(?=.*$p)" for p in patterns), "i")
end

gpu(value::Real, patterns...) =
    GPURule(gpu_regex(patterns), () -> value)

gpu(f::F, patterns...) where {F} =
    GPURule(gpu_regex(patterns), f)


const PEAK_FLOPS = (
    # NVIDIA Blackwell
    gpu(2.5e15,  "gb200"),
    gpu(2.5e15,  "grace blackwell"),
    gpu(2.25e15, "b200"),
    gpu(1.8e15,  "b100"),

    # NVIDIA Hopper
    gpu(836e12, "h200", "nvl"),
    gpu(836e12, "h200", "pcie"),
    gpu(989e12, "h200"),
    gpu(835e12, "h100", "nvl"),
    gpu(756e12, "h100", "pcie"),
    gpu(989e12, "h100"),
    gpu(989e12, "h800", "nvl"),
    gpu(756e12, "h800"),

    # NVIDIA Ampere
    gpu(312e12,   "a100"),
    gpu(312e12,   "a800"),
    gpu(149.7e12, "a40"),
    gpu(165e12,   "a30"),

    # NVIDIA Ada
    gpu(362e12, "l40s"),
    gpu(362e12, "l40-s"),
    gpu(362e12, "l40 s"),
    gpu(121e12, "l4"),

    # AMD CDNA
    gpu(2.5e15,    "mi355"),
    gpu(1.3074e15, "mi325"),
    gpu(1.3074e15, "mi300x"),
    gpu(980.6e12,  "mi300a"),
    gpu(383e12,    "mi250x"),
    gpu(362.1e12,  "mi250"),

    # Consumer RTX
    gpu(209.5e12, "5090"),
    gpu(165.2e12, "4090"),
    gpu(71e12,    "3090"),

    # Intel Ponte Vecchio
    gpu("data center gpu max 1550") do
        props = XPU.device_properties()
        512 * props.max_compute_units * 1300e6
    end,
)

const PEAK_BANDWIDTH = (
    # NVIDIA Blackwell
    gpu(8.0e12, "gb200"),
    gpu(8.0e12, "grace blackwell"),
    gpu(8.0e12, "b200"),
    gpu(8.0e12, "b100"),

    # NVIDIA Hopper
    gpu(4.8e12,  "h200"),
    gpu(3.9e12,  "h100", "nvl"),
    gpu(2.0e12,  "h100", "pcie"),
    gpu(3.35e12, "h100"),
    gpu(2.0e12,  "h800", "pcie"),
    gpu(3.35e12, "h800"),

    # NVIDIA Ampere
    gpu(2.0e12, "a100"),
    gpu(2.0e12, "a800"),
    gpu(696e9,  "a40"),
    gpu(933e9,  "a30"),

    # NVIDIA Ada
    gpu(864e9, "l40s"),
    gpu(864e9, "l40-s"),
    gpu(864e9, "l40 s"),
    gpu(300e9, "l4"),

    # AMD CDNA
    gpu(8.0e12,  "mi355"),
    gpu(6.0e12,  "mi325"),
    gpu(5.3e12,  "mi300x"),
    gpu(5.3e12,  "mi300a"),
    gpu(3.28e12, "mi250x"),
    gpu(3.28e12, "mi250"),

    # Consumer RTX
    gpu(1.79e12, "5090"),
    gpu(1.01e12, "4090"),
    gpu(936e9,   "3090"),
)


function gpu_value(name, rules, default=-Inf)
    for rule in rules
        occursin(rule.pattern, name) && return rule.value()
    end

    return default
end

get_peak_flops(name::AbstractString) = gpu_value(name, PEAK_FLOPS, -Inf)

get_peak_bandwidth(name::AbstractString) = gpu_value(name, PEAK_BANDWIDTH; default=Inf)

end