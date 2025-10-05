Temperature Enhanced

A high-performance Python implementation for robot DH parameter calibration using Hamiltonian Monte Carlo (HMC) with temperature-based exploration.
Built with PyMC, NumPyro, and JAX, it estimates d1, a2, a3, and a4 with full uncertainty quantification and temperature-controlled global sampling.

🔧 Features

Turbo-optimized HMC with multi-core CPU parallelization

Temperature-enhanced inference: Parallel Tempering + Simulated Annealing

NumPyro NUTS sampler for efficient gradient-based MCMC

Vectorized forward kinematics (Modified DH convention)

Full statistical diagnostics — ESS, R-hat, HDI, and relative errors

Automatic temperature analysis and plotting

🧠 Methods

This script combines Bayesian inference and robot kinematics to estimate Denavit–Hartenberg parameters from noisy end-effector data.
Temperature scaling improves exploration of multimodal posteriors, reducing the risk of local minima during calibration.

📊 Outputs

Temperature vs parameter evolution plots

Relative error trends and HDI coverage

Comprehensive calibration report printed to console

⚙️ Requirements

Install dependencies:

pip install pymc numpyro jax arviz matplotlib seaborn scipy

🚀 Usage

Run the main script:

python turbo_hmc_dh_calibration.py


Choose between:

Parallel Tempering

Simulated Annealing

or both (comparison mode)

Results include best temperature selection, credible intervals, and visualization of parameter convergence.

📈 Example Applications

Industrial robot calibration

Kinematic model refinement

Bayesian inference experiments in robotics or physics

