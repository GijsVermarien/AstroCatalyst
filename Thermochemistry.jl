using Catalyst
using DifferentialEquations
using Plots
using Symbolics


# Some basic astrochemistry constants:
# u_vec = [H2 O C O⁺ OH⁺ H H2O⁺ H3O⁺ E H2O OH C⁺ CO CO⁺ H⁺ HCO⁺ T]
# println(u_vec)
# @species 
kboltzmann = 1.38064852e-16  # erg / K
pmass = 1.6726219e-24  # g
# dust2gas = 1e-2 # ratio
mu = 2.34
seconds_per_year = 3600 * 24 * 365
gamma_ad = 1.4
gnot = 1e0
# Simulation parameters:
number_density = 1e4
dust2gas = 0.01
minimum_fractional_density = 1e-30 * number_density

@register_symbolic get_heating(H, H2, E, tgas, ntot)
function get_heating(H, H2, E, tgas, ntot)
    """
       get_heating(x, tgas, cr_rate, gnot)

    Calculate the total heating rate based on various processes.

    ## Arguments
    - `x`: Dict{String, Float64} — A dictionary containing the abundances of different species:
        - `"H"`: Abundance of hydrogen
        - `"H2"`: Abundance of molecular hydrogen
        - `"E"`: Abundance of electrons
        - `"dust2gas"`: Dust-to-gas ratio
    - `tgas`: Float64 — Gas temperature
    - `cr_rate`: Float64 — Cosmic ray ionization rate
    - `gnot`: Float64 — Scaling factor for cosmic ray ionization rate

    ## Returns
    - Float64 — Total heating rate considering cosmic ray ionization and photoelectric heating processes.
    """

    rate_H2 = 5.68e-11 * gnot
    heats = [
        cosmic_ionisation_rate * (5.5e-12 * H + 2.5e-11 * H2),
        get_photoelectric_heating(H, E, tgas, gnot, ntot),
        6.4e-13 * rate_H2 * H2,
    ]

    return sum(heats)
end

@register_symbolic get_photoelectric_heating(H, E, tgas, gnot, ntot)
function get_photoelectric_heating(H, E, tgas, gnot, ntot)
    """
       get_photoelectric_heating(x, tgas, gnot)

    Calculate the photoelectric heating rate due to dust grains.

    ## Arguments
    - `x`: Dict{String, Float64} — A dictionary containing the abundances of different species:
        - `"H"`: Abundance of hydrogen
        - `"H2"`: Abundance of molecular hydrogen
        - `"E"`: Abundance of electrons
    - `tgas`: Float64 — Gas temperature
    - `gnot`: Float64 — Scaling factor for cosmic ray ionization rate

    ## Returns
    - Float64 — Photoelectric heating rate based on dust recombination and ionization processes.
    """
    # ntot = sum(x)
    bet = 0.735 * tgas^(-0.068)
    psi = (E>0) * gnot * sqrt(tgas) / E

    # grains recombination cooling
    recomb_cool = 4.65e-30 * tgas^0.94 * psi^bet * E * H

    eps = 4.9e-2 / (1 + 4e-3 * psi^0.73) + 3.7e-2 * (tgas * 1e-4)^0.7 / (1 + 2e-4 * psi)

    # net photoelectric heating
    return (1.3e-24 * eps * gnot * ntot - recomb_cool) * dust2gas
end

@register_symbolic get_cooling(H, H2, O, E, tgas)
function get_cooling(H, H2, O, E, tgas)
    """
       get_cooling(x, tgas)

    Calculate the total cooling rate based on various processes.

    ## Arguments
    - `x`: Dict{String, Float64} — A dictionary containing the abundances of different species:
        - `"H"`: Abundance of hydrogen
        - `"E"`: Abundance of electrons
        - `"O"`: Abundance of oxygen
        - `"H2"`: Abundance of molecular hydrogen
    - `tgas`: Float64 — Gas temperature

    ## Returns
    - Float64 — Total cooling rate considering Lyman-alpha, OI 630nm, and H2 cooling processes.
    """

    cool = 7.3e-19 * H * E * exp(-118400.0 / tgas)  # Ly-alpha
    cool += 1.8e-24 * O * E * exp(-22800 / tgas)  # OI 630nm
    cool += cooling_H2(H, H2, tgas) # H2 cooling by dissacoiation and recombination
    return cool
end

@register_symbolic cooling_H2(H, H2, temp)
function cooling_H2(H, H2, temp)
    """
       cooling_H2(x, temp)

    Calculate the cooling rate for molecular hydrogen (H2) at a given temperature.

    ## Arguments
    - `x`: Dict{String, Float64} — A dictionary containing the abundances of different species:
        - `"H"`: Abundance of hydrogen
        - `"H2"`: Abundance of molecular hydrogen
    - `temp`: Float64 — Gas temperature

    ## Returns
    - Float64 — Cooling rate due to molecular hydrogen (H2) dissociation and recombination processes.
    """
    t3 = temp * 1e-3  # (T/1000)
    logt3 = log10(t3)

    logt32 = logt3 * logt3
    logt33 = logt32 * logt3
    logt34 = logt33 * logt3
    logt35 = logt34 * logt3
    logt36 = logt35 * logt3
    logt37 = logt36 * logt3
    logt38 = logt37 * logt3

    if temp < 2e3
        HDLR = (9.5e-22 * t3^3.76) / (1.0 + 0.12 * t3^2.1) * exp(-((0.13 / t3)^3)) + 3.0e-24 * exp(-0.51 / t3)
        HDLV = 6.7e-19 * exp(-5.86 / t3) + 1.6e-18 * exp(-11.7 / t3)
        HDL = HDLR + HDLV
    elseif 2e3 <= temp <= 1e4
        HDL = 1e1^(
            -2.0584225e1
            +
            5.0194035 * logt3
            -
            1.5738805 * logt32
            -
            4.7155769 * logt33
            + 2.4714161 * logt34
            + 5.4710750 * logt35
            -
            3.9467356 * logt36
            -
            2.2148338 * logt37
            +
            1.8161874 * logt38
        )
    else
        HDL = 5.531333679406485e-19
    end

    if temp <= 1e2
        f = 1e1^(
            -16.818342e0
            + 3.7383713e1 * logt3
            + 5.8145166e1 * logt32
            + 4.8656103e1 * logt33
            + 2.0159831e1 * logt34
            + 3.8479610e0 * logt35
        )
    elseif 1e2 < temp <= 1e3
        f = 1e1^(
            -2.4311209e1
            +
            3.5692468e0 * logt3
            -
            1.1332860e1 * logt32
            -
            2.7850082e1 * logt33
            -
            2.1328264e1 * logt34
            -
            4.2519023e0 * logt35
        )
    elseif 1e3 < temp <= 6e3
        f = 1e1^(
            -2.4311209e1
            +
            4.6450521e0 * logt3
            -
            3.7209846e0 * logt32
            +
            5.9369081e0 * logt33
            -
            5.5108049e0 * logt34
            +
            1.5538288e0 * logt35
        )
    else
        f = 1.862314467912518e-22
    end

    LDL = f * H

    if LDL * HDL == 0.0
        return 0.0
    end

    cool = H2 / (1.0 / HDL + 1.0 / LDL)

    return cool
end

function get_heating_cooling(T, H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺)
    ntot = get_ntot(H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺)
    return (gamma_ad - 1e0) * (get_heating(H, H2, E, T, ntot) - get_cooling(H, H2, O, E, T)) / kboltzmann / ntot
end

function get_ntot(H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺)
    return sum([H2 O C O⁺ OH⁺ H H2O⁺ H3O⁺ E H2O OH C⁺ CO CO⁺ H⁺ HCO⁺])
end

ka_reaction(Tgas, α=1.0, β=1.0, γ=0.0) = α*(Tgas/300)^β*exp(−γ / Tgas)


# CONTINUE HERE
# Try this: https://docs.sciml.ai/Catalyst/stable/catalyst_functionality/constraint_equations/#Coupling-ODE-constraints-via-directly-building-a-ReactionSystem


@variables t T(t) = 100.0 # Define the variables before the species!
@species H2(t) O(t) C(t) O⁺(t) OH⁺(t) H(t) H2O⁺(t) H3O⁺(t) E(t) H2O(t) OH(t) C⁺(t) CO(t) CO⁺(t) H⁺(t) HCO⁺(t)
@parameters cosmic_ionisation_rate radiation_field dust2gas

D = Differential(t)
reaction_equations = [
	(@reaction 1.6e-9, $O⁺ + $H2 --> $OH⁺ + $H),
	(@reaction 1e-9, $OH⁺ + $H2 --> $H2O⁺ + $H),
	(@reaction 6.1e-10, $H2O⁺ + $H2 --> $H3O⁺ + $H),
	(@reaction ka_reaction(T, 1.1e-7, -1/2), $H3O⁺ + $E --> $H2O + $H),
	(@reaction ka_reaction(T, 8.6e-8, -1/2), $H2O⁺ + $E --> $OH + $H),
	(@reaction ka_reaction(T, 3.9e-8, -1/2), $H2O⁺ + $E --> $O + $H2),
	(@reaction ka_reaction(T, 6.3e-9, -0.48), $OH⁺ + $E --> $O + $H),
	(@reaction ka_reaction(T, 3.4e-12, -0.63), $O⁺ + $E --> $O),
	(@reaction 2.8 * cosmic_ionisation_rate, $O --> $O⁺ + $E),
	(@reaction 2.62 * cosmic_ionisation_rate, $C --> $C⁺ + $E),
	(@reaction 5.0 * cosmic_ionisation_rate, $CO --> $C + $O),
	(@reaction ka_reaction(T, 4.4e-12, -0.61), $C⁺ + $E --> $C),
	(@reaction ka_reaction(T, 1.15e-10, -0.339), $C⁺ + $OH --> CO + $H),
	(@reaction 9.15e-10 * (0.62 + 0.4767 * 5.5 * sqrt(300 / T)), $C⁺ + $OH --> $CO⁺ + $H),
	(@reaction 4e-10, $CO⁺ + $H --> $CO + $H⁺),
	(@reaction 7.28e-10, $CO⁺ + $H2 --> $HCO⁺ + $H),
	(@reaction ka_reaction(T, 2.8e-7, -0.69), $HCO⁺ + $E --> $CO + $H),
	(@reaction ka_reaction(T, 3.5e-12, -0.7), $H⁺ + $E --> $H),
	(@reaction 2.121e-17 * dust2gas / 1e-2, $H + $H --> $H2),
    (@reaction 1e-1 * cosmic_ionisation_rate, $H2 --> $H + $H),
	(@reaction 3.39e-10 * radiation_field, $C --> $C⁺ + $E),
	(@reaction 2.43e-10 * radiation_field, $CO --> $C + $O),
	(@reaction 7.72e-10 * radiation_field, $H2O --> $OH + $H),
    (D(T) ~ get_heating_cooling(T, H2, O, C, O⁺, OH⁺, H, H2O⁺, H3O⁺, E, H2O, OH, C⁺, CO, CO⁺, H⁺, HCO⁺)) 
]

@named system = ReactionSystem(reaction_equations, t)

u0 = [:H2 => number_density, :O => number_density*2e-4, :C => number_density*1e-4, :O⁺=>minimum_fractional_density, :OH⁺=>minimum_fractional_density, :H=> minimum_fractional_density, :H2O⁺=> minimum_fractional_density, :H3O⁺=>minimum_fractional_density, :E=>minimum_fractional_density, :H2O=>minimum_fractional_density, :OH=>minimum_fractional_density, :C⁺=>minimum_fractional_density, :CO=>minimum_fractional_density, :CO⁺=>minimum_fractional_density, :H⁺=>minimum_fractional_density, :HCO⁺=> minimum_fractional_density, :T=> 100.0]

odesys = convert(ODESystem, system)

setdefaults!(system, u0)

tspan = (0.0, 1e6*seconds_per_year)

params = [dust2gas => 0.01, radiation_field => 1e-1, cosmic_ionisation_rate => 1e-17]

println("Lets try to solve the ODE:")

sys = convert(ODESystem,system)
oprob = clipboard(ODEProblemExpr(sys, [], tspan, params))

oprob = ODEProblem(system, [], tspan, params)
println("Created the ODEproblem.")
sol = solve(oprob, Tsit5())
println("Solved the ODE")
# println(sol)

# plot(sol)
# plot!(xcsale=:log10, yscale=:log10)
# plot!(legend=:outerbottom, legendcolumns=3)

# a = Array(sol)
# plot(a[17, :])

using LinearAlgebra
prob = begin
    #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:1165 =#
    f = begin
            #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:743 =#
            var"##f#525" = (ModelingToolkit.ODEFunctionClosure)(function (ˍ₋arg1, ˍ₋arg2, t)
                        #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:373 =#
                        #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:374 =#
                        #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:375 =#
                        begin
                            begin
                                #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:468 =#
                                (SymbolicUtils.Code.create_array)(typeof(ˍ₋arg1), nothing, Val{1}(), Val{(17,)}(), (+)((+)((/)((*)((*)(-3.4e-12, ˍ₋arg1[7]), ˍ₋arg1[1]), (*)(0.027505124147434282, (^)(ˍ₋arg2[1], 0.63))), (*)((*)(-1.6e-9, ˍ₋arg1[2]), ˍ₋arg1[1])), (*)((*)(2.8, ˍ₋arg2[2]), ˍ₋arg1[10])), (+)((+)((+)((+)((+)((+)((/)((*)((*)(3.9e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (*)((*)(1.0605e-15, ˍ₋arg2[3]), (^)(ˍ₋arg1[4], 2))), (*)((*)(-6.1e-10, ˍ₋arg1[2]), ˍ₋arg1[5])), (*)((*)(-0.1, ˍ₋arg2[2]), ˍ₋arg1[2])), (*)((*)(-7.28e-10, ˍ₋arg1[14]), ˍ₋arg1[2])), (*)((*)(-1.0e-9, ˍ₋arg1[2]), ˍ₋arg1[3])), (*)((*)(-1.6e-9, ˍ₋arg1[2]), ˍ₋arg1[1])), (+)((+)((/)((*)((*)(-6.3e-9, ˍ₋arg1[7]), ˍ₋arg1[3]), (*)(0.06471154931041326, (^)(ˍ₋arg2[1], 0.48))), (*)((*)(-1.0e-9, ˍ₋arg1[2]), ˍ₋arg1[3])), (*)((*)(1.6e-9, ˍ₋arg1[2]), ˍ₋arg1[1])), (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((*)(8.6e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (/)((*)((*)(3.5e-12, ˍ₋arg1[7]), ˍ₋arg1[15]), (*)(0.01845079661875638, (^)(ˍ₋arg2[1], 0.7)))), (/)((*)((*)(1.15e-10, ˍ₋arg1[12]), ˍ₋arg1[9]), (*)(0.14462917025834096, (^)(ˍ₋arg2[1], 0.339)))), (/)((*)((*)(6.3e-9, ˍ₋arg1[7]), ˍ₋arg1[3]), (*)(0.06471154931041326, (^)(ˍ₋arg2[1], 0.48)))), (/)((*)((*)(1.1e-7, ˍ₋arg1[7]), ˍ₋arg1[6]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (/)((*)((*)(2.8e-7, ˍ₋arg1[7]), ˍ₋arg1[16]), (*)(0.019533781893248173, (^)(ˍ₋arg2[1], 0.69)))), (*)((*)(-2.121e-15, ˍ₋arg2[3]), (^)(ˍ₋arg1[4], 2))), (*)((*)(7.72e-10, ˍ₋arg2[4]), ˍ₋arg1[8])), (*)((*)(-4.0e-10, ˍ₋arg1[14]), ˍ₋arg1[4])), (*)((*)(6.1e-10, ˍ₋arg1[2]), ˍ₋arg1[5])), (*)((*)(0.2, ˍ₋arg2[2]), ˍ₋arg1[2])), (*)((*)(7.28e-10, ˍ₋arg1[14]), ˍ₋arg1[2])), (*)((*)(1.0e-9, ˍ₋arg1[2]), ˍ₋arg1[3])), (*)((*)(1.6e-9, ˍ₋arg1[2]), ˍ₋arg1[1])), (*)((*)((*)(9.15e-10, (+)(0.62, (*)(2.6218500000000002, (sqrt)((/)(300, ˍ₋arg2[1]))))), ˍ₋arg1[12]), ˍ₋arg1[9])), (+)((+)((+)((/)((*)((*)(-3.9e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (/)((*)((*)(-8.6e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (*)((*)(-6.1e-10, ˍ₋arg1[2]), ˍ₋arg1[5])), (*)((*)(1.0e-9, ˍ₋arg1[2]), ˍ₋arg1[3])), (+)((/)((*)((*)(-1.1e-7, ˍ₋arg1[7]), ˍ₋arg1[6]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (*)((*)(6.1e-10, ˍ₋arg1[2]), ˍ₋arg1[5])), (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((*)(-4.4e-12, ˍ₋arg1[12]), ˍ₋arg1[7]), (*)(0.030828758425176017, (^)(ˍ₋arg2[1], 0.61))), (/)((*)((*)(-3.9e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (/)((*)((*)(-8.6e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (/)((*)((*)(-1.1e-7, ˍ₋arg1[7]), ˍ₋arg1[6]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (/)((*)((*)(-2.8e-7, ˍ₋arg1[7]), ˍ₋arg1[16]), (*)(0.019533781893248173, (^)(ˍ₋arg2[1], 0.69)))), (/)((*)((*)(-3.4e-12, ˍ₋arg1[7]), ˍ₋arg1[1]), (*)(0.027505124147434282, (^)(ˍ₋arg2[1], 0.63)))), (/)((*)((*)(-3.5e-12, ˍ₋arg1[7]), ˍ₋arg1[15]), (*)(0.01845079661875638, (^)(ˍ₋arg2[1], 0.7)))), (/)((*)((*)(-6.3e-9, ˍ₋arg1[7]), ˍ₋arg1[3]), (*)(0.06471154931041326, (^)(ˍ₋arg2[1], 0.48)))), (*)((*)(3.39e-10, ˍ₋arg2[4]), ˍ₋arg1[11])), (*)((*)(2.62, ˍ₋arg2[2]), ˍ₋arg1[11])), (*)((*)(2.8, ˍ₋arg2[2]), ˍ₋arg1[10])), (+)((/)((*)((*)(1.1e-7, ˍ₋arg1[7]), ˍ₋arg1[6]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (*)((*)(-7.72e-10, ˍ₋arg2[4]), ˍ₋arg1[8])), (+)((+)((+)((/)((*)((*)(8.6e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (/)((*)((*)(-1.15e-10, ˍ₋arg1[12]), ˍ₋arg1[9]), (*)(0.14462917025834096, (^)(ˍ₋arg2[1], 0.339)))), (*)((*)(7.72e-10, ˍ₋arg2[4]), ˍ₋arg1[8])), (*)((*)((*)(-9.15e-10, (+)(0.62, (*)(2.6218500000000002, (sqrt)((/)(300, ˍ₋arg2[1]))))), ˍ₋arg1[12]), ˍ₋arg1[9])), (+)((+)((+)((+)((+)((/)((*)((*)(3.4e-12, ˍ₋arg1[7]), ˍ₋arg1[1]), (*)(0.027505124147434282, (^)(ˍ₋arg2[1], 0.63))), (/)((*)((*)(6.3e-9, ˍ₋arg1[7]), ˍ₋arg1[3]), (*)(0.06471154931041326, (^)(ˍ₋arg2[1], 0.48)))), (/)((*)((*)(3.9e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (*)((*)(2.43e-10, ˍ₋arg2[4]), ˍ₋arg1[13])), (*)((*)(5.0, ˍ₋arg2[2]), ˍ₋arg1[13])), (*)((*)(-2.8, ˍ₋arg2[2]), ˍ₋arg1[10])), (+)((+)((+)((+)((/)((*)((*)(4.4e-12, ˍ₋arg1[12]), ˍ₋arg1[7]), (*)(0.030828758425176017, (^)(ˍ₋arg2[1], 0.61))), (*)((*)(-3.39e-10, ˍ₋arg2[4]), ˍ₋arg1[11])), (*)((*)(2.43e-10, ˍ₋arg2[4]), ˍ₋arg1[13])), (*)((*)(-2.62, ˍ₋arg2[2]), ˍ₋arg1[11])), (*)((*)(5.0, ˍ₋arg2[2]), ˍ₋arg1[13])), (+)((+)((+)((+)((/)((*)((*)(-4.4e-12, ˍ₋arg1[12]), ˍ₋arg1[7]), (*)(0.030828758425176017, (^)(ˍ₋arg2[1], 0.61))), (/)((*)((*)(-1.15e-10, ˍ₋arg1[12]), ˍ₋arg1[9]), (*)(0.14462917025834096, (^)(ˍ₋arg2[1], 0.339)))), (*)((*)(3.39e-10, ˍ₋arg2[4]), ˍ₋arg1[11])), (*)((*)(2.62, ˍ₋arg2[2]), ˍ₋arg1[11])), (*)((*)((*)(-9.15e-10, (+)(0.62, (*)(2.6218500000000002, (sqrt)((/)(300, ˍ₋arg2[1]))))), ˍ₋arg1[12]), ˍ₋arg1[9])), (+)((+)((+)((+)((/)((*)((*)(2.8e-7, ˍ₋arg1[7]), ˍ₋arg1[16]), (*)(0.019533781893248173, (^)(ˍ₋arg2[1], 0.69))), (/)((*)((*)(1.15e-10, ˍ₋arg1[12]), ˍ₋arg1[9]), (*)(0.14462917025834096, (^)(ˍ₋arg2[1], 0.339)))), (*)((*)(-2.43e-10, ˍ₋arg2[4]), ˍ₋arg1[13])), (*)((*)(4.0e-10, ˍ₋arg1[14]), ˍ₋arg1[4])), (*)((*)(-5.0, ˍ₋arg2[2]), ˍ₋arg1[13])), (+)((+)((*)((*)(-4.0e-10, ˍ₋arg1[14]), ˍ₋arg1[4]), (*)((*)(-7.28e-10, ˍ₋arg1[14]), ˍ₋arg1[2])), (*)((*)((*)(9.15e-10, (+)(0.62, (*)(2.6218500000000002, (sqrt)((/)(300, ˍ₋arg2[1]))))), ˍ₋arg1[12]), ˍ₋arg1[9])), (+)((/)((*)((*)(-3.5e-12, ˍ₋arg1[7]), ˍ₋arg1[15]), (*)(0.01845079661875638, (^)(ˍ₋arg2[1], 0.7))), (*)((*)(4.0e-10, ˍ₋arg1[14]), ˍ₋arg1[4])), (+)((/)((*)((*)(-2.8e-7, ˍ₋arg1[7]), ˍ₋arg1[16]), (*)(0.019533781893248173, (^)(ˍ₋arg2[1], 0.69))), (*)((*)(7.28e-10, ˍ₋arg1[14]), ˍ₋arg1[2])), (/)((*)(2.897189213660258e15, (+)((*)(-1, (get_cooling)(ˍ₋arg1[4], ˍ₋arg1[2], ˍ₋arg1[10], ˍ₋arg1[7], ˍ₋arg1[17])), (get_heating)(ˍ₋arg1[4], ˍ₋arg1[2], ˍ₋arg1[7], ˍ₋arg1[17], (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)(ˍ₋arg1[11], ˍ₋arg1[13]), ˍ₋arg1[14]), ˍ₋arg1[12]), ˍ₋arg1[7]), ˍ₋arg1[4]), ˍ₋arg1[2]), ˍ₋arg1[8]), ˍ₋arg1[5]), ˍ₋arg1[6]), ˍ₋arg1[16]), ˍ₋arg1[15]), ˍ₋arg1[10]), ˍ₋arg1[9]), ˍ₋arg1[3]), ˍ₋arg1[1])))), (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)(ˍ₋arg1[11], ˍ₋arg1[13]), ˍ₋arg1[14]), ˍ₋arg1[12]), ˍ₋arg1[7]), ˍ₋arg1[4]), ˍ₋arg1[2]), ˍ₋arg1[8]), ˍ₋arg1[5]), ˍ₋arg1[6]), ˍ₋arg1[16]), ˍ₋arg1[15]), ˍ₋arg1[10]), ˍ₋arg1[9]), ˍ₋arg1[3]), ˍ₋arg1[1])))
                            end
                        end
                    end, function (ˍ₋out, ˍ₋arg1, ˍ₋arg2, t)
                        #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:373 =#
                        #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:374 =#
                        #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:375 =#
                        begin
                            begin
                                #= /Users/gijsv/.julia/packages/Symbolics/3ueSK/src/build_function.jl:537 =#
                                #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:422 =# @inbounds begin
                                        #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:418 =#
                                        ˍ₋out[1] = (+)((+)((/)((*)((*)(-3.4e-12, ˍ₋arg1[7]), ˍ₋arg1[1]), (*)(0.027505124147434282, (^)(ˍ₋arg2[1], 0.63))), (*)((*)(-1.6e-9, ˍ₋arg1[2]), ˍ₋arg1[1])), (*)((*)(2.8, ˍ₋arg2[2]), ˍ₋arg1[10]))
                                        ˍ₋out[2] = (+)((+)((+)((+)((+)((+)((/)((*)((*)(3.9e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (*)((*)(1.0605e-15, ˍ₋arg2[3]), (^)(ˍ₋arg1[4], 2))), (*)((*)(-6.1e-10, ˍ₋arg1[2]), ˍ₋arg1[5])), (*)((*)(-0.1, ˍ₋arg2[2]), ˍ₋arg1[2])), (*)((*)(-7.28e-10, ˍ₋arg1[14]), ˍ₋arg1[2])), (*)((*)(-1.0e-9, ˍ₋arg1[2]), ˍ₋arg1[3])), (*)((*)(-1.6e-9, ˍ₋arg1[2]), ˍ₋arg1[1]))
                                        ˍ₋out[3] = (+)((+)((/)((*)((*)(-6.3e-9, ˍ₋arg1[7]), ˍ₋arg1[3]), (*)(0.06471154931041326, (^)(ˍ₋arg2[1], 0.48))), (*)((*)(-1.0e-9, ˍ₋arg1[2]), ˍ₋arg1[3])), (*)((*)(1.6e-9, ˍ₋arg1[2]), ˍ₋arg1[1]))
                                        ˍ₋out[4] = (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((*)(8.6e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (/)((*)((*)(3.5e-12, ˍ₋arg1[7]), ˍ₋arg1[15]), (*)(0.01845079661875638, (^)(ˍ₋arg2[1], 0.7)))), (/)((*)((*)(1.15e-10, ˍ₋arg1[12]), ˍ₋arg1[9]), (*)(0.14462917025834096, (^)(ˍ₋arg2[1], 0.339)))), (/)((*)((*)(6.3e-9, ˍ₋arg1[7]), ˍ₋arg1[3]), (*)(0.06471154931041326, (^)(ˍ₋arg2[1], 0.48)))), (/)((*)((*)(1.1e-7, ˍ₋arg1[7]), ˍ₋arg1[6]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (/)((*)((*)(2.8e-7, ˍ₋arg1[7]), ˍ₋arg1[16]), (*)(0.019533781893248173, (^)(ˍ₋arg2[1], 0.69)))), (*)((*)(-2.121e-15, ˍ₋arg2[3]), (^)(ˍ₋arg1[4], 2))), (*)((*)(7.72e-10, ˍ₋arg2[4]), ˍ₋arg1[8])), (*)((*)(-4.0e-10, ˍ₋arg1[14]), ˍ₋arg1[4])), (*)((*)(6.1e-10, ˍ₋arg1[2]), ˍ₋arg1[5])), (*)((*)(0.2, ˍ₋arg2[2]), ˍ₋arg1[2])), (*)((*)(7.28e-10, ˍ₋arg1[14]), ˍ₋arg1[2])), (*)((*)(1.0e-9, ˍ₋arg1[2]), ˍ₋arg1[3])), (*)((*)(1.6e-9, ˍ₋arg1[2]), ˍ₋arg1[1])), (*)((*)((*)(9.15e-10, (+)(0.62, (*)(2.6218500000000002, (sqrt)((/)(300, ˍ₋arg2[1]))))), ˍ₋arg1[12]), ˍ₋arg1[9]))
                                        ˍ₋out[5] = (+)((+)((+)((/)((*)((*)(-3.9e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (/)((*)((*)(-8.6e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (*)((*)(-6.1e-10, ˍ₋arg1[2]), ˍ₋arg1[5])), (*)((*)(1.0e-9, ˍ₋arg1[2]), ˍ₋arg1[3]))
                                        ˍ₋out[6] = (+)((/)((*)((*)(-1.1e-7, ˍ₋arg1[7]), ˍ₋arg1[6]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (*)((*)(6.1e-10, ˍ₋arg1[2]), ˍ₋arg1[5]))
                                        ˍ₋out[7] = (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((*)(-4.4e-12, ˍ₋arg1[12]), ˍ₋arg1[7]), (*)(0.030828758425176017, (^)(ˍ₋arg2[1], 0.61))), (/)((*)((*)(-3.9e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (/)((*)((*)(-8.6e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (/)((*)((*)(-1.1e-7, ˍ₋arg1[7]), ˍ₋arg1[6]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (/)((*)((*)(-2.8e-7, ˍ₋arg1[7]), ˍ₋arg1[16]), (*)(0.019533781893248173, (^)(ˍ₋arg2[1], 0.69)))), (/)((*)((*)(-3.4e-12, ˍ₋arg1[7]), ˍ₋arg1[1]), (*)(0.027505124147434282, (^)(ˍ₋arg2[1], 0.63)))), (/)((*)((*)(-3.5e-12, ˍ₋arg1[7]), ˍ₋arg1[15]), (*)(0.01845079661875638, (^)(ˍ₋arg2[1], 0.7)))), (/)((*)((*)(-6.3e-9, ˍ₋arg1[7]), ˍ₋arg1[3]), (*)(0.06471154931041326, (^)(ˍ₋arg2[1], 0.48)))), (*)((*)(3.39e-10, ˍ₋arg2[4]), ˍ₋arg1[11])), (*)((*)(2.62, ˍ₋arg2[2]), ˍ₋arg1[11])), (*)((*)(2.8, ˍ₋arg2[2]), ˍ₋arg1[10]))
                                        ˍ₋out[8] = (+)((/)((*)((*)(1.1e-7, ˍ₋arg1[7]), ˍ₋arg1[6]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (*)((*)(-7.72e-10, ˍ₋arg2[4]), ˍ₋arg1[8]))
                                        ˍ₋out[9] = (+)((+)((+)((/)((*)((*)(8.6e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5))), (/)((*)((*)(-1.15e-10, ˍ₋arg1[12]), ˍ₋arg1[9]), (*)(0.14462917025834096, (^)(ˍ₋arg2[1], 0.339)))), (*)((*)(7.72e-10, ˍ₋arg2[4]), ˍ₋arg1[8])), (*)((*)((*)(-9.15e-10, (+)(0.62, (*)(2.6218500000000002, (sqrt)((/)(300, ˍ₋arg2[1]))))), ˍ₋arg1[12]), ˍ₋arg1[9]))
                                        ˍ₋out[10] = (+)((+)((+)((+)((+)((/)((*)((*)(3.4e-12, ˍ₋arg1[7]), ˍ₋arg1[1]), (*)(0.027505124147434282, (^)(ˍ₋arg2[1], 0.63))), (/)((*)((*)(6.3e-9, ˍ₋arg1[7]), ˍ₋arg1[3]), (*)(0.06471154931041326, (^)(ˍ₋arg2[1], 0.48)))), (/)((*)((*)(3.9e-8, ˍ₋arg1[7]), ˍ₋arg1[5]), (*)(0.05773502691896258, (^)(ˍ₋arg2[1], 0.5)))), (*)((*)(2.43e-10, ˍ₋arg2[4]), ˍ₋arg1[13])), (*)((*)(5.0, ˍ₋arg2[2]), ˍ₋arg1[13])), (*)((*)(-2.8, ˍ₋arg2[2]), ˍ₋arg1[10]))
                                        ˍ₋out[11] = (+)((+)((+)((+)((/)((*)((*)(4.4e-12, ˍ₋arg1[12]), ˍ₋arg1[7]), (*)(0.030828758425176017, (^)(ˍ₋arg2[1], 0.61))), (*)((*)(-3.39e-10, ˍ₋arg2[4]), ˍ₋arg1[11])), (*)((*)(2.43e-10, ˍ₋arg2[4]), ˍ₋arg1[13])), (*)((*)(-2.62, ˍ₋arg2[2]), ˍ₋arg1[11])), (*)((*)(5.0, ˍ₋arg2[2]), ˍ₋arg1[13]))
                                        ˍ₋out[12] = (+)((+)((+)((+)((/)((*)((*)(-4.4e-12, ˍ₋arg1[12]), ˍ₋arg1[7]), (*)(0.030828758425176017, (^)(ˍ₋arg2[1], 0.61))), (/)((*)((*)(-1.15e-10, ˍ₋arg1[12]), ˍ₋arg1[9]), (*)(0.14462917025834096, (^)(ˍ₋arg2[1], 0.339)))), (*)((*)(3.39e-10, ˍ₋arg2[4]), ˍ₋arg1[11])), (*)((*)(2.62, ˍ₋arg2[2]), ˍ₋arg1[11])), (*)((*)((*)(-9.15e-10, (+)(0.62, (*)(2.6218500000000002, (sqrt)((/)(300, ˍ₋arg2[1]))))), ˍ₋arg1[12]), ˍ₋arg1[9]))
                                        ˍ₋out[13] = (+)((+)((+)((+)((/)((*)((*)(2.8e-7, ˍ₋arg1[7]), ˍ₋arg1[16]), (*)(0.019533781893248173, (^)(ˍ₋arg2[1], 0.69))), (/)((*)((*)(1.15e-10, ˍ₋arg1[12]), ˍ₋arg1[9]), (*)(0.14462917025834096, (^)(ˍ₋arg2[1], 0.339)))), (*)((*)(-2.43e-10, ˍ₋arg2[4]), ˍ₋arg1[13])), (*)((*)(4.0e-10, ˍ₋arg1[14]), ˍ₋arg1[4])), (*)((*)(-5.0, ˍ₋arg2[2]), ˍ₋arg1[13]))
                                        ˍ₋out[14] = (+)((+)((*)((*)(-4.0e-10, ˍ₋arg1[14]), ˍ₋arg1[4]), (*)((*)(-7.28e-10, ˍ₋arg1[14]), ˍ₋arg1[2])), (*)((*)((*)(9.15e-10, (+)(0.62, (*)(2.6218500000000002, (sqrt)((/)(300, ˍ₋arg2[1]))))), ˍ₋arg1[12]), ˍ₋arg1[9]))
                                        ˍ₋out[15] = (+)((/)((*)((*)(-3.5e-12, ˍ₋arg1[7]), ˍ₋arg1[15]), (*)(0.01845079661875638, (^)(ˍ₋arg2[1], 0.7))), (*)((*)(4.0e-10, ˍ₋arg1[14]), ˍ₋arg1[4]))
                                        ˍ₋out[16] = (+)((/)((*)((*)(-2.8e-7, ˍ₋arg1[7]), ˍ₋arg1[16]), (*)(0.019533781893248173, (^)(ˍ₋arg2[1], 0.69))), (*)((*)(7.28e-10, ˍ₋arg1[14]), ˍ₋arg1[2]))
                                        ˍ₋out[17] = @show (/)((*)(2.897189213660258e15, (+)((*)(-1, (get_cooling)(ˍ₋arg1[4], ˍ₋arg1[2], ˍ₋arg1[10], ˍ₋arg1[7], ˍ₋arg1[17])), (get_heating)(ˍ₋arg1[4], ˍ₋arg1[2], ˍ₋arg1[7], ˍ₋arg1[17], (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)(ˍ₋arg1[11], ˍ₋arg1[13]), ˍ₋arg1[14]), ˍ₋arg1[12]), ˍ₋arg1[7]), ˍ₋arg1[4]), ˍ₋arg1[2]), ˍ₋arg1[8]), ˍ₋arg1[5]), ˍ₋arg1[6]), ˍ₋arg1[16]), ˍ₋arg1[15]), ˍ₋arg1[10]), ˍ₋arg1[9]), ˍ₋arg1[3]), ˍ₋arg1[1])))), (+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)(ˍ₋arg1[11], ˍ₋arg1[13]), ˍ₋arg1[14]), ˍ₋arg1[12]), ˍ₋arg1[7]), ˍ₋arg1[4]), ˍ₋arg1[2]), ˍ₋arg1[8]), ˍ₋arg1[5]), ˍ₋arg1[6]), ˍ₋arg1[16]), ˍ₋arg1[15]), ˍ₋arg1[10]), ˍ₋arg1[9]), ˍ₋arg1[3]), ˍ₋arg1[1]))
                                        #= /Users/gijsv/.julia/packages/SymbolicUtils/Oyu8Z/src/code.jl:420 =#
                                        nothing
                                    end
                            end
                        end
                    end)
            #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:744 =#
            var"##tgrad#526" = nothing
            #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:745 =#
            var"##jac#527" = nothing
            #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:746 =#
            M = LinearAlgebra.UniformScaling{Bool}(true)
            #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:747 =#
            ODEFunction{true}(var"##f#525", jac = var"##jac#527", tgrad = var"##tgrad#526", mass_matrix = M, jac_prototype = nothing, syms = [Symbol("O⁺(t)"), Symbol("H2(t)"), Symbol("OH⁺(t)"), Symbol("H(t)"), Symbol("H2O⁺(t)"), Symbol("H3O⁺(t)"), Symbol("E(t)"), Symbol("H2O(t)"), Symbol("OH(t)"), Symbol("O(t)"), Symbol("C(t)"), Symbol("C⁺(t)"), Symbol("CO(t)"), Symbol("CO⁺(t)"), Symbol("H⁺(t)"), Symbol("HCO⁺(t)"), Symbol("T(t)")], indepsym = :t, paramsyms = [:T, :cosmic_ionisation_rate, :dust2gas, :radiation_field], sparsity = nothing, observed = nothing)
        end
    #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:1166 =#
    u0 = [1.0e-26, 10000.0, 1.0e-26, 1.0e-26, 1.0e-26, 1.0e-26, 1.0e-26, 1.0e-26, 1.0e-26, 2.0, 1.0, 1.0e-26, 1.0e-26, 1.0e-26, 1.0e-26, 1.0e-26, 100.0]
    #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:1167 =#
    tspan = (0.0, 3.1536e13)
    #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:1168 =#
    p = [100.0, 1.0e-17, 0.01, 0.1]
    #= /Users/gijsv/.julia/packages/ModelingToolkit/BsHty/src/systems/diffeqs/abstractodesystem.jl:1169 =#
    ODEProblem(f, u0, tspan, p; )
end

sol = solve(prob, Tsit5())