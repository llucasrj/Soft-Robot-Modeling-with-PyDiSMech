from dismech.params import Material

aisi_304_stainless_steel = Material(                            
    density=8000,
    youngs_rod=193e9,
    youngs_shell=193e9,
    poisson_rod=0.29,
    poisson_shell=0.29
)

aluminum_6061 = Material(
    density=2700,
    youngs_rod=69e9,
    youngs_shell=69e9,
    poisson_rod=0.33,
    poisson_shell=0.33
)

silicone_rubber = Material(
    density=1100,   
    youngs_rod=1e6,
    youngs_shell=1e6,
    poisson_rod=0.49,
    poisson_shell=0.49
)

carbon_fiber_composite = Material(
    density=1600,  
    youngs_rod=70e9,
    youngs_shell=70e9,
    poisson_rod=0.3,
    poisson_shell=0.3
)
