from src import (
        build_qubo, brute_force_optimal, synthetic_instance,
        find_optimal_params, make_grid_circuits, run_grid, build_energy_grid,
        plot_landscape, plot_approximation_ratios,
    )
import numpy as np, dataclasses

Q, mu, sigma, lam = synthetic_instance(n=6, k=3, seed=42)
e_opt, x_opt = brute_force_optimal(Q, 3)
print(f'e_opt: {e_opt:.4f}')
