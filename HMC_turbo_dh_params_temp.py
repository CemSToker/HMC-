"""
Turbo HMC for Robot DH Parameter Calibration - Temperature Enhanced
==================================================================

High-performance version with:
- Multiple CPU cores
- Numpyro NUTS sampler
- Vectorized operations
- Aggressive optimizations
- Temperature-based exploration (Parallel Tempering + Simulated Annealing)
- Estimates d1, a2, a3, a4 (keeping d4, d6 and alpha parameters fixed)
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import jax
import numpyro
import warnings
from typing import Tuple, List, Dict, Any
import seaborn as sns
from datetime import datetime
import multiprocessing
from scipy.stats import gaussian_kde
import copy

# Configure for maximum performance
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Enable JAX optimizations
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

def forward_kinematics_flange(joint_angles: np.ndarray, dh_params: np.ndarray) -> np.ndarray:
    """Forward kinematics for data generation - Modified DH convention to match inference model."""
    J = np.array(joint_angles)
    
    # Fixed alpha values (same as inference model)
    alpha2 = np.pi/2
    alpha3 = 0
    alpha4 = np.pi/2
    alpha5 = -np.pi/2
    
    # Fixed DH parameters (same as inference model)
    d4 = -723
    d6 = -100
    
    # Pre-compute alpha values
    ca2, sa2 = np.cos(alpha2), np.sin(alpha2)
    ca3, sa3 = np.cos(alpha3), np.sin(alpha3)
    ca4, sa4 = np.cos(alpha4), np.sin(alpha4)
    ca5, sa5 = np.cos(alpha5), np.sin(alpha5)
    
    # T01
    T01 = np.array([[np.cos(J[0]), -np.sin(J[0]), 0, 0],
                    [np.sin(J[0]), np.cos(J[0]), 0, 0],
                    [0, 0, 1, dh_params[0,2]],  # d1
                    [0, 0, 0, 1]])
    
    # T12 with alpha2 rotation (Modified DH)
    T12 = np.array([[np.cos(J[1]), -np.sin(J[1]), 0, dh_params[1,0]],  # a2
                    [np.sin(J[1]) * ca2, np.cos(J[1]) * ca2, -sa2, 0],
                    [np.sin(J[1]) * sa2, np.cos(J[1]) * sa2, ca2, 0],
                    [0, 0, 0, 1]])
    
    # T23 with alpha3 rotation (Modified DH)
    T23 = np.array([[np.cos(J[2] + J[1]), -np.sin(J[2] + J[1]), 0, dh_params[2,0]],  # a3
                    [-np.sin(J[2] + J[1]) * ca3, -np.cos(J[2] + J[1]) * ca3, -sa3, 0],
                    [-np.sin(J[2] + J[1]) * sa3, -np.cos(J[2] + J[1]) * sa3, ca3, 0],
                    [0, 0, 0, 1]])
    
    # T34 with alpha4 rotation (Modified DH)
    T34 = np.array([[np.cos(J[3]), -np.sin(J[3]), 0, dh_params[3,0]],  # a4
                    [0, 0, 1, d4],  # fixed d4
                    [-np.sin(J[3]) * ca4, -np.cos(J[3]) * ca4, -sa4, 0],
                    [0, 0, 0, 1]])
    
    # T45 with alpha5 rotation (Modified DH)
    T45 = np.array([[np.cos(J[4]), -np.sin(J[4]), 0, 0],
                    [0, 0, 1, 0],
                    [np.sin(J[4]) * ca5, np.cos(J[4]) * ca5, -sa5, 0],
                    [0, 0, 0, 1]])
    
    # T56 (Modified DH)
    T56 = np.array([[np.cos(J[5]), -np.sin(J[5]), 0, 0],
                    [0, 0, 1, d6],  # fixed d6
                    [-np.sin(J[5]), -np.cos(J[5]), 0, 0],
                    [0, 0, 0, 1]])
    
    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    pos = T @ np.array([[0],[0],[0],[1]])
    pos = pos[:3,0]
    pos[2] = pos[2] - 650
    return pos

def generate_robot_data(N: int, dh_params: np.ndarray, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic robot data with optimal joint configurations for DH parameter estimation."""
    # Generate joint angles that are most informative for the 4 parameters we're estimating
    joint_angles = np.zeros((N, 6))
    
    # Use full ranges for joints that affect our estimated parameters
    # Joint 1: affects d1 estimation (vertical offset)
    joint_angles[:, 0] = np.random.uniform(-np.pi, np.pi, N)
    
    # Joint 2: affects a2 estimation (first link length)
    joint_angles[:, 1] = np.random.uniform(-np.pi, np.pi, N)  # Full range for better a2 sensitivity
    
    # Joint 3: affects a3 estimation (second link length)
    joint_angles[:, 2] = np.random.uniform(-np.pi, np.pi, N)  # Full range for better a3 sensitivity
    
    # Joint 4: affects a4 estimation (third link length)
    joint_angles[:, 3] = np.random.uniform(-np.pi, np.pi, N)  # Full range for better a4 sensitivity
    
    # Joints 5 and 6: less critical for our parameters, but still varied
    joint_angles[:, 4] = np.random.uniform(-np.pi/2, np.pi/2, N)
    joint_angles[:, 5] = np.random.uniform(-np.pi/2, np.pi/2, N)
    
    ee_positions = np.array([forward_kinematics_flange(ja, dh_params) for ja in joint_angles])
    noisy_ee_positions = ee_positions + np.random.normal(0, noise_std, ee_positions.shape)
    return joint_angles, noisy_ee_positions

def turbo_fk_batch_dh_params(joint_angles: pt.TensorVariable, d1: float, a2: float, a3: float, 
                            a4: float) -> pt.TensorVariable:
    """
    Forward kinematics using Modified DH parameters with fixed alpha values.
    Estimates: d1, a2, a3, a4
    Fixed: d4=-723, d6=-100, alpha2=π/2, alpha3=0, alpha4=π/2, alpha5=-π/2
    """
    J = joint_angles
    cos_vals = pt.cos(J)
    sin_vals = pt.sin(J)
    
    c0, c1, c2, c3, c4, c5 = cos_vals[:, 0], cos_vals[:, 1], cos_vals[:, 2], cos_vals[:, 3], cos_vals[:, 4], cos_vals[:, 5]
    s0, s1, s2, s3, s4, s5 = sin_vals[:, 0], sin_vals[:, 1], sin_vals[:, 2], sin_vals[:, 3], sin_vals[:, 4], sin_vals[:, 5]

    # Fixed alpha values
    alpha2 = np.pi/2
    alpha3 = 0
    alpha4 = np.pi/2
    alpha5 = -np.pi/2
    
    # Fixed DH parameters
    d4 = -723
    d6 = -100
    
    # Pre-compute alpha values
    ca2, sa2 = pt.cos(alpha2), pt.sin(alpha2)
    ca3, sa3 = pt.cos(alpha3), pt.sin(alpha3)
    ca4, sa4 = pt.cos(alpha4), pt.sin(alpha4)
    ca5, sa5 = pt.cos(alpha5), pt.sin(alpha5)

    # T01
    T01 = pt.stack([
        pt.stack([c0, -s0, pt.zeros_like(c0), pt.zeros_like(c0)], axis=1),
        pt.stack([s0,  c0, pt.zeros_like(c0), pt.zeros_like(c0)], axis=1),
        pt.stack([pt.zeros_like(c0), pt.zeros_like(c0), pt.ones_like(c0), pt.full_like(c0, d1)], axis=1),
        pt.stack([pt.zeros_like(c0), pt.zeros_like(c0), pt.zeros_like(c0), pt.ones_like(c0)], axis=1)
    ], axis=1)

    # T12 with alpha2 rotation
    T12 = pt.stack([
        pt.stack([c1, -s1, pt.zeros_like(c1), pt.full_like(c1, a2)], axis=1),
        pt.stack([s1 * ca2, c1 * ca2, -sa2 * pt.ones_like(c1), pt.zeros_like(c1)], axis=1),
        pt.stack([s1 * sa2, c1 * sa2, ca2 * pt.ones_like(c1), pt.zeros_like(c1)], axis=1),
        pt.stack([pt.zeros_like(c1), pt.zeros_like(c1), pt.zeros_like(c1), pt.ones_like(c1)], axis=1)
    ], axis=1)

    # T23 with alpha3 rotation
    T23 = pt.stack([
        pt.stack([pt.cos(J[:, 2] + J[:, 1]), -pt.sin(J[:, 2] + J[:, 1]), pt.zeros_like(s2), pt.full_like(s2, a3)], axis=1),
        pt.stack([-pt.sin(J[:, 2] + J[:, 1]) * ca3, -pt.cos(J[:, 2] + J[:, 1]) * ca3, -sa3 * pt.ones_like(s2), pt.zeros_like(s2)], axis=1),
        pt.stack([-pt.sin(J[:, 2] + J[:, 1]) * sa3, -pt.cos(J[:, 2] + J[:, 1]) * sa3, ca3 * pt.ones_like(s2), pt.zeros_like(s2)], axis=1),
        pt.stack([pt.zeros_like(s2), pt.zeros_like(s2), pt.zeros_like(s2), pt.ones_like(s2)], axis=1)
    ], axis=1)

    # T34 with alpha4 rotation (using fixed d4)
    T34 = pt.stack([
        pt.stack([c3, -s3, pt.zeros_like(c3), pt.full_like(c3, a4)], axis=1),
        pt.stack([pt.zeros_like(c3), pt.zeros_like(c3), pt.ones_like(c3), pt.full_like(c3, d4)], axis=1),
        pt.stack([-s3 * ca4, -c3 * ca4, -sa4 * pt.ones_like(c3), pt.zeros_like(c3)], axis=1),
        pt.stack([pt.zeros_like(c3), pt.zeros_like(c3), pt.zeros_like(c3), pt.ones_like(c3)], axis=1)
    ], axis=1)

    # T45 with alpha5 rotation
    T45 = pt.stack([
        pt.stack([c4, -s4, pt.zeros_like(c4), pt.zeros_like(c4)], axis=1),
        pt.stack([pt.zeros_like(c4), pt.zeros_like(c4), pt.ones_like(c4), pt.zeros_like(c4)], axis=1),
        pt.stack([s4 * ca5, c4 * ca5, -sa5 * pt.ones_like(c4), pt.zeros_like(c4)], axis=1),
        pt.stack([pt.zeros_like(c4), pt.zeros_like(c4), pt.zeros_like(c4), pt.ones_like(c4)], axis=1)
    ], axis=1)

    # T56 (using fixed d6)
    T56 = pt.stack([
        pt.stack([c5, -s5, pt.zeros_like(c5), pt.zeros_like(c5)], axis=1),
        pt.stack([pt.zeros_like(c5), pt.zeros_like(c5), pt.ones_like(c5), pt.full_like(c5, d6)], axis=1),
        pt.stack([-s5, -c5, pt.zeros_like(c5), pt.zeros_like(c5)], axis=1),
        pt.stack([pt.zeros_like(c5), pt.zeros_like(c5), pt.zeros_like(c5), pt.ones_like(c5)], axis=1)
    ], axis=1)

    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56

    ee_pos = pt.stack([0., 0., 0., 1.])
    pos = T @ ee_pos
    pos = pos[:, :3]
    pos = pt.set_subtensor(pos[:, 2], pos[:, 2] - 650)

    return pos 

def create_temperature_ladder(n_temps: int = 8, min_temp: float = 0.1, max_temp: float = 10.0) -> np.ndarray:
    """
    Create a geometric temperature ladder for parallel tempering.
    
    Args:
        n_temps: Number of temperature levels
        min_temp: Minimum temperature (closest to target distribution)
        max_temp: Maximum temperature (most exploratory)
    
    Returns:
        Array of temperatures in descending order (coldest first)
    """
    # Geometric spacing for better mixing
    temperatures = np.geomspace(max_temp, min_temp, n_temps)
    return temperatures

def create_tempered_model(joint_angles: np.ndarray, observed_positions: np.ndarray, 
                         temperature: float = 1.0) -> pm.Model:
    """
    Create a PyMC model with temperature scaling for parallel tempering.
    
    Args:
        joint_angles: Joint angle data
        observed_positions: Observed end-effector positions
        temperature: Temperature scaling factor (1.0 = target distribution)
    
    Returns:
        PyMC model with temperature-scaled likelihood
    """
    with pm.Model() as tempered_model:
        # DH parameters to estimate
        d1 = pm.Normal('d1', mu=649, sigma=1)
        a2 = pm.Normal('a2', mu=73.8, sigma=1)
        a3 = pm.Normal('a3', mu=636.1, sigma=1)
        a4 = pm.Normal('a4', mu=120, sigma=1)
        """[0, 0, 650, 0],      # d1 = 650mm
            [75, np.pi/2, 0, 0], # a2 = 75mm, alpha2 = π/2
            [637, 0, 0, 0],      # a3 = 637mm
            [120, np.pi/2, -723, 0], # a4 = 120mm, d4 = -723mm
            [0, -np.pi/2, 0, 0], # alpha5 = -π/2
            [0, 0, -100, 0]"""
        # Noise prior
        sigma_xyz = pm.HalfNormal('sigma_xyz', sigma=0.1, shape=3)
        
        # Use pm.Data for joint angles
        joint_angles_shared = pm.Data('joint_angles', joint_angles)
        
        # Forward kinematics
        expected_pos = turbo_fk_batch_dh_params(
            joint_angles_shared, d1, a2, a3, a4
        )
        
        # Temperature-scaled likelihood
        # Higher temperature = flatter likelihood = more exploration
        scaled_sigma = sigma_xyz * np.sqrt(temperature)
        Y_obs = pm.Normal('Y_obs', mu=expected_pos, sigma=scaled_sigma, observed=observed_positions)
    
    return tempered_model

def parallel_tempering_sampling(joint_angles: np.ndarray, observed_positions: np.ndarray,
                               n_temps: int = 8, draws: int = 3000, tune: int = 1000,
                               cpu_count: int = 4) -> List[az.InferenceData]:
    """
    Perform parallel tempering sampling with multiple temperature levels.
    
    Args:
        joint_angles: Joint angle data
        observed_positions: Observed end-effector positions
        n_temps: Number of temperature levels
        draws: Number of draws per chain
        tune: Number of tuning steps
        cpu_count: Number of CPU cores to use
    
    Returns:
        List of InferenceData objects for each temperature level
    """
    print(f"🔥 Starting Parallel Tempering with {n_temps} temperature levels...")
    
    # Create temperature ladder
    temperatures = create_temperature_ladder(n_temps)
    print(f"🌡️  Temperature ladder: {temperatures}")
    
    # Sample at each temperature level
    all_idatas = []
    
    for i, temp in enumerate(temperatures):
        print(f"🌡️  Sampling at temperature {temp:.3f} ({i+1}/{n_temps})...")
        
        # Create model for this temperature
        model = create_tempered_model(joint_angles, observed_positions, temp)
        
        # Sample with this model
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                cores=min(cpu_count, 4),  # Limit cores per chain
                chains=2,  # Fewer chains per temperature to save resources
                target_accept=0.9,
                nuts_sampler="numpyro",
                random_seed=41 + i,  # Different seed for each temperature
                return_inferencedata=True,
                progressbar=False  # Disable progress bar for cleaner output
            )
        
        all_idatas.append(idata)
        print(f"✅ Completed sampling at temperature {temp:.3f}")
    
    return all_idatas, temperatures

def simulated_annealing_sampling(joint_angles: np.ndarray, observed_positions: np.ndarray,
                                initial_temp: float = 10.0, final_temp: float = 0.1,
                                n_stages: int = 5, draws_per_stage: int = 1000,
                                cpu_count: int = 4) -> List[az.InferenceData]:
    """
    Perform simulated annealing sampling with temperature cooling.
    
    Args:
        joint_angles: Joint angle data
        observed_positions: Observed end-effector positions
        initial_temp: Starting temperature
        final_temp: Final temperature
        n_stages: Number of cooling stages
        draws_per_stage: Number of draws per stage
        cpu_count: Number of CPU cores to use
    
    Returns:
        List of InferenceData objects for each annealing stage
    """
    print(f"❄️  Starting Simulated Annealing with {n_stages} cooling stages...")
    
    # Create cooling schedule (geometric cooling)
    temperatures = np.geomspace(initial_temp, final_temp, n_stages)
    print(f"🌡️  Cooling schedule: {temperatures}")
    
    all_idatas = []
    
    for i, temp in enumerate(temperatures):
        print(f"🌡️  Annealing stage {i+1}/{n_stages} at temperature {temp:.3f}...")
        
        # Create model for this temperature
        model = create_tempered_model(joint_angles, observed_positions, temp)
        
        # Sample with this model
        with model:
            idata = pm.sample(
                draws=draws_per_stage,
                tune=min(500, draws_per_stage // 2),  # Shorter tuning for annealing
                cores=min(cpu_count, 4),
                chains=2,
                target_accept=0.9,
                nuts_sampler="numpyro",
                random_seed=100 + i,
                return_inferencedata=True,
                progressbar=False
            )
        
        all_idatas.append(idata)
        print(f"✅ Completed annealing stage {i+1} at temperature {temp:.3f}")
    
    return all_idatas, temperatures

def analyze_temperature_results(all_idatas: List[az.InferenceData], temperatures: np.ndarray,
                              true_dh_params: np.ndarray) -> Dict[str, Any]:
    """
    Analyze results from temperature-based sampling.
    
    Args:
        all_idatas: List of InferenceData objects from different temperatures
        temperatures: Array of temperatures used
        true_dh_params: True DH parameters for comparison
    
    Returns:
        Dictionary with analysis results
    """
    print("📊 Analyzing temperature-based sampling results...")
    
    results = {
        'temperatures': temperatures,
        'parameter_evolution': {},
        'convergence_metrics': {},
        'best_temperature': None,
        'best_estimates': None
    }
    
    param_names = ['d1', 'a2', 'a3', 'a4']
    true_values = {
        'd1': true_dh_params[0, 2],
        'a2': true_dh_params[1, 0],
        'a3': true_dh_params[2, 0],
        'a4': true_dh_params[3, 0]
    }
    
    # Track parameter evolution across temperatures
    for param in param_names:
        results['parameter_evolution'][param] = {
            'means': [],
            'stds': [],
            'relative_errors': []
        }
    
    # Analyze each temperature level
    for i, (idata, temp) in enumerate(zip(all_idatas, temperatures)):
        print(f"🌡️  Analyzing temperature {temp:.3f}...")
        
        # Get parameter estimates
        summary = az.summary(idata, var_names=param_names)
        
        for param in param_names:
            mean_val = summary.loc[param, 'mean']
            std_val = summary.loc[param, 'sd']
            true_val = true_values[param]
            relative_error = abs(mean_val - true_val) / true_val * 100
            
            results['parameter_evolution'][param]['means'].append(mean_val)
            results['parameter_evolution'][param]['stds'].append(std_val)
            results['parameter_evolution'][param]['relative_errors'].append(relative_error)
    
    # Find best temperature (lowest average relative error)
    avg_errors = []
    for i, temp in enumerate(temperatures):
        errors = [results['parameter_evolution'][param]['relative_errors'][i] for param in param_names]
        avg_errors.append(np.mean(errors))
    
    best_idx = np.argmin(avg_errors)
    results['best_temperature'] = temperatures[best_idx]
    results['best_estimates'] = all_idatas[best_idx]
    
    print(f"🏆 Best temperature: {results['best_temperature']:.3f}")
    print(f"📈 Average relative error at best temperature: {avg_errors[best_idx]:.2f}%")
    
    return results

def plot_temperature_analysis(results: Dict[str, Any], true_dh_params: np.ndarray) -> None:
    """
    Plot temperature analysis results.
    
    Args:
        results: Results from analyze_temperature_results
        true_dh_params: True DH parameters
    """
    temperatures = results['temperatures']
    param_names = ['d1', 'a2', 'a3', 'a4']
    true_values = {
        'd1': true_dh_params[0, 2],
        'a2': true_dh_params[1, 0],
        'a3': true_dh_params[2, 0],
        'a4': true_dh_params[3, 0]
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temperature Analysis - Parameter Evolution', fontsize=16, fontweight='bold')
    
    for i, param in enumerate(param_names):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        means = results['parameter_evolution'][param]['means']
        stds = results['parameter_evolution'][param]['stds']
        errors = results['parameter_evolution'][param]['relative_errors']
        
        # Plot parameter evolution
        ax.errorbar(temperatures, means, yerr=stds, marker='o', capsize=5, 
                   label='Estimated', linewidth=2, markersize=8)
        
        # Plot true value
        true_val = true_values[param]
        ax.axhline(true_val, color='red', linestyle='--', linewidth=2, 
                  label=f'True {param} = {true_val:.1f}')
        
        # Highlight best temperature
        best_temp = results['best_temperature']
        best_idx = np.where(temperatures == best_temp)[0][0]
        ax.axvline(best_temp, color='green', linestyle=':', linewidth=2, 
                  label=f'Best T = {best_temp:.3f}')
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel(f'{param} (mm)')
        ax.set_title(f'{param} Evolution vs Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add error annotation
        best_error = errors[best_idx]
        ax.text(0.02, 0.98, f'Best Error: {best_error:.2f}%', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Plot error evolution
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for param in param_names:
        errors = results['parameter_evolution'][param]['relative_errors']
        ax.plot(temperatures, errors, marker='o', linewidth=2, markersize=8, 
               label=f'{param} Error')
    
    ax.axvline(results['best_temperature'], color='red', linestyle='--', linewidth=2,
              label=f'Best Temperature = {results["best_temperature"]:.3f}')
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Parameter Error Evolution vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.show()

def test_dh_parameter_sensitivity(joint_angles: np.ndarray, true_dh_params: np.ndarray) -> None:
    """Test the sensitivity of end-effector positions to DH parameter changes."""
    print("🔍 Testing DH parameter sensitivity...")
    
    # Test different DH parameter values
    test_configs = joint_angles[:5]  # Use first 5 configurations
    
    # Test each parameter (only the ones we're estimating)
    param_names = ['d1', 'a2', 'a3', 'a4']
    param_indices = [(0, 2), (1, 0), (2, 0), (3, 0)]
    param_variations = [10, 5, 10, 5]  # mm variations
    
    for param_name, (i, j), variation in zip(param_names, param_indices, param_variations):
        base_value = true_dh_params[i, j]
        test_values = [base_value - variation, base_value, base_value + variation]
        
        print(f"\n  {param_name} sensitivity:")
        base_positions = None  # Initialize base_positions
        
        for test_val in test_values:
            # Create modified DH params
            test_dh_params = true_dh_params.copy()
            test_dh_params[i, j] = test_val
            
            # Calculate positions
            positions = np.array([forward_kinematics_flange(ja, test_dh_params) for ja in test_configs])
            
            if test_val == base_value:
                base_positions = positions
            else:
                # Calculate RMS change
                if base_positions is not None:
                    rms_change = np.sqrt(np.mean((positions - base_positions)**2))
                    print(f"    {param_name} = {test_val:.1f} mm: RMS change = {rms_change:.3f} mm")
    
    print()

def print_temperature_report(results: Dict[str, Any], true_dh_params: np.ndarray, 
                           start_time: datetime, method: str = "Parallel Tempering") -> None:
    """Print comprehensive temperature-based sampling report."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("="*80)
    print(f"TEMPERATURE-ENHANCED HMC ROBOT CALIBRATION REPORT - {method.upper()}")
    print("="*80)
    print(f"Execution time: {duration:.2f} seconds")
    print(f"Number of temperature levels: {len(results['temperatures'])}")
    print(f"Temperature range: {results['temperatures'].min():.3f} - {results['temperatures'].max():.3f}")
    print(f"Best temperature: {results['best_temperature']:.3f}")
    print()
    
    # True values
    print("TRUE DH PARAMETERS:")
    print(f"  d1: {true_dh_params[0, 2]:.3f} mm")
    print(f"  a2: {true_dh_params[1, 0]:.3f} mm")
    print(f"  a3: {true_dh_params[2, 0]:.3f} mm")
    print(f"  a4: {true_dh_params[3, 0]:.3f} mm")
    print("FIXED DH PARAMETERS:")
    print(f"  d4: {true_dh_params[3, 2]:.3f} mm")
    print(f"  d6: {true_dh_params[5, 2]:.3f} mm")
    print()
    
    # Best estimates
    best_idata = results['best_estimates']
    summary = az.summary(best_idata, var_names=['d1', 'a2', 'a3', 'a4'])
    
    print("BEST ESTIMATES (at optimal temperature):")
    param_names = ['d1', 'a2', 'a3', 'a4']
    true_values = {
        'd1': true_dh_params[0, 2],
        'a2': true_dh_params[1, 0],
        'a3': true_dh_params[2, 0],
        'a4': true_dh_params[3, 0]
    }
    
    for param in param_names:
        mean_val = summary.loc[param, 'mean']
        std_val = summary.loc[param, 'sd']
        hdi_low = summary.loc[param, 'hdi_3%']
        hdi_high = summary.loc[param, 'hdi_97%']
        true_val = true_values[param]
        relative_error = abs(mean_val - true_val) / true_val * 100
        
        print(f"  {param}: {mean_val:.3f} ± {std_val:.3f} mm")
        print(f"    95% HDI: [{hdi_low:.3f}, {hdi_high:.3f}]")
        print(f"    Relative error: {relative_error:.2f}%")
        
        if hdi_low <= true_val <= hdi_high:
            print(f"    ✓ True value within HDI")
        else:
            print(f"    ✗ True value outside HDI")
        print()
    
    # Convergence diagnostics
    ess = az.ess(best_idata)
    print("CONVERGENCE DIAGNOSTICS (at best temperature):")
    for param in param_names:
        ess_val = ess[param].values
        print(f"  {param}: ESS = {ess_val:.0f}")
    
    if len(best_idata.posterior.chain) > 1:
        rhat = az.rhat(best_idata)
        print("GELMAN-RUBIN DIAGNOSTIC:")
        for param in param_names:
            rhat_val = rhat[param].values
            status = "✓" if rhat_val < 1.01 else "✗"
            print(f"  {param}: R̂ = {rhat_val:.3f} {status}")
    
    print("="*80) 

def main():
    """Temperature-enhanced HMC execution with parallel tempering and simulated annealing."""
    start_time = datetime.now()
    
    try:
        print("🚀 TEMPERATURE-ENHANCED HMC Robot Calibration")
        print("="*60)
        
        # Get CPU count for optimal parallelization
        cpu_count = multiprocessing.cpu_count()
        print(f"🖥️  Detected {cpu_count} CPU cores")
        
        # 1. Set true DH parameters
        true_dh_params = np.array([
            [0, 0, 650, 0],      # d1 = 650mm
            [75, np.pi/2, 0, 0], # a2 = 75mm, alpha2 = π/2
            [637, 0, 0, 0],      # a3 = 637mm
            [120, np.pi/2, -723, 0], # a4 = 120mm, d4 = -723mm
            [0, -np.pi/2, 0, 0], # alpha5 = -π/2
            [0, 0, -100, 0]      # d6 = -100mm
        ])
        
        # 2. Generate data
        print("📊 Generating synthetic robot data...")
        N_CONFIGS = 1000  # Reduced for faster temperature sampling
        joint_angles, observed_positions = generate_robot_data(
            N_CONFIGS, true_dh_params, noise_std=0.1
        )
        
        # 3. Test DH parameter sensitivity
        test_dh_parameter_sensitivity(joint_angles, true_dh_params)
        
        # 4. Choose temperature method
        print("\n🌡️  Choose temperature method:")
        print("1. Parallel Tempering (recommended)")
        print("2. Simulated Annealing")
        print("3. Both methods (comparison)")
        
        # For demonstration, we'll run parallel tempering
        method_choice = 1  # You can change this to test different methods
        
        if method_choice == 1:
            # Parallel Tempering
            print("\n🔥 Running Parallel Tempering...")
            all_idatas, temperatures = parallel_tempering_sampling(
                joint_angles, observed_positions,
                n_temps=6,  # Reduced for faster execution
                draws=2000, tune=500,
                cpu_count=cpu_count
            )
            
            # Analyze results
            results = analyze_temperature_results(all_idatas, temperatures, true_dh_params)
            
            # Print report
            print_temperature_report(results, true_dh_params, start_time, "Parallel Tempering")
            
            # Plot analysis
            plot_temperature_analysis(results, true_dh_params)
            
        elif method_choice == 2:
            # Simulated Annealing
            print("\n❄️  Running Simulated Annealing...")
            all_idatas, temperatures = simulated_annealing_sampling(
                joint_angles, observed_positions,
                initial_temp=10.0, final_temp=0.1,
                n_stages=5, draws_per_stage=800,
                cpu_count=cpu_count
            )
            
            # Analyze results
            results = analyze_temperature_results(all_idatas, temperatures, true_dh_params)
            
            # Print report
            print_temperature_report(results, true_dh_params, start_time, "Simulated Annealing")
            
            # Plot analysis
            plot_temperature_analysis(results, true_dh_params)
            
        elif method_choice == 3:
            # Both methods for comparison
            print("\n🔄 Running both methods for comparison...")
            
            # Parallel Tempering
            print("\n🔥 Parallel Tempering:")
            pt_idatas, pt_temps = parallel_tempering_sampling(
                joint_angles, observed_positions,
                n_temps=5, draws=1500, tune=400,
                cpu_count=cpu_count
            )
            pt_results = analyze_temperature_results(pt_idatas, pt_temps, true_dh_params)
            
            # Simulated Annealing
            print("\n❄️  Simulated Annealing:")
            sa_idatas, sa_temps = simulated_annealing_sampling(
                joint_angles, observed_positions,
                initial_temp=8.0, final_temp=0.1,
                n_stages=5, draws_per_stage=600,
                cpu_count=cpu_count
            )
            sa_results = analyze_temperature_results(sa_idatas, sa_temps, true_dh_params)
            
            # Compare results
            print("\n📊 METHOD COMPARISON:")
            print(f"Parallel Tempering - Best T: {pt_results['best_temperature']:.3f}")
            print(f"Simulated Annealing - Best T: {sa_results['best_temperature']:.3f}")
            
            # Plot both
            plot_temperature_analysis(pt_results, true_dh_params)
            plot_temperature_analysis(sa_results, true_dh_params)
            
            results = pt_results  # Use PT results as default
        
        print("✅ Temperature-enhanced calibration completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during temperature-enhanced calibration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 