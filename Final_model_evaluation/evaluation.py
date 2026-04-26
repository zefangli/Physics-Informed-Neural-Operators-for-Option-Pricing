import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader
import time
from scipy.stats import binned_statistic_2d

# Import your classes from the training script
# Ensure your training script is named 'training.py' in the same folder
from training import DeepBS_Solver, ChronologicalOptionDataset, create_pinn_collate_fn

def analytical_black_scholes(M, tau, r, q, sigma):
    """
    Computes the Black-Scholes formula for a European Call Option.
    Since the network predicts v = C/K (price normalized by strike), 
    we substitute S/K with M (Moneyness).
    """
    # Handle the edge case where tau is very close to 0 to avoid division by zero
    tau = np.maximum(tau, 1e-8)
    
    d1 = (np.log(M) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    # C/K = M * e^(-q*tau) * N(d1) - e^(-r*tau) * N(d2)
    v_analytical = M * np.exp(-q * tau) * norm.cdf(d1) - np.exp(-r * tau) * norm.cdf(d2)
    
    # Intrinsic value floor (options can't be worth less than intrinsic value at expiration)
    intrinsic = np.maximum(M - 1.0, 0.0)
    v_analytical = np.where(tau <= 1e-8, intrinsic, v_analytical)
    
    return v_analytical


def plot_analytical_sweeps(model, device):
    print("Generating Analytical vs Network 2 Sweeps...")
    model.eval()
    
    # Number of points for the sweep
    N = 100
    
    # Baseline fixed parameters
    M_fixed = 1.0     # At-the-money
    tau_fixed = 1.0   # 1 year to maturity
    r_fixed = 0.05    # 5% risk-free rate
    q_fixed = 0.0     # 0% dividend yield
    sigma_fixed = 0.2 # 20% volatility
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ---------------------------------------------------------
    # Sweep 1: Varying Moneyness (M)
    # ---------------------------------------------------------
    M_sweep = np.linspace(0.5, 1.5, N).astype(np.float32)
    
    inputs_1 = torch.tensor(np.column_stack([
        M_sweep, 
        np.full(N, tau_fixed), 
        np.full(N, r_fixed), 
        np.full(N, q_fixed), 
        np.full(N, sigma_fixed)
    ]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        v_hat_1 = model.nn2(inputs_1).cpu().numpy().flatten()
        
    v_true_1 = analytical_black_scholes(M_sweep, tau_fixed, r_fixed, q_fixed, sigma_fixed)
    
    axes[0].plot(M_sweep, v_true_1, 'k-', lw=2, label='Analytical BS')
    axes[0].scatter(M_sweep, v_hat_1, c='red', s=15, label='PINN_MLP Predict', zorder=3)
    axes[0].set_title(f'Varying Moneyness (M)\n$\\tau$={tau_fixed}, r={r_fixed}, $\\sigma$={sigma_fixed}')
    axes[0].set_xlabel('Moneyness (M = S/K)')
    axes[0].set_ylabel('Normalized Price (v = C/K)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # ---------------------------------------------------------
    # Sweep 2: Varying Time to Maturity (tau)
    # ---------------------------------------------------------
    tau_sweep = np.linspace(0.01, 3.0, N).astype(np.float32)
    
    inputs_2 = torch.tensor(np.column_stack([
        np.full(N, M_fixed), 
        tau_sweep, 
        np.full(N, r_fixed), 
        np.full(N, q_fixed), 
        np.full(N, sigma_fixed)
    ]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        v_hat_2 = model.nn2(inputs_2).cpu().numpy().flatten()
        
    v_true_2 = analytical_black_scholes(M_fixed, tau_sweep, r_fixed, q_fixed, sigma_fixed)
    
    axes[1].plot(tau_sweep, v_true_2, 'k-', lw=2, label='Analytical BS')
    axes[1].scatter(tau_sweep, v_hat_2, c='red', s=15, label='PINN_MLP Predict', zorder=3)
    axes[1].set_title(f'Varying Time ($\\tau$)\nM={M_fixed}, r={r_fixed}, $\\sigma$={sigma_fixed}')
    axes[1].set_xlabel('Time to Maturity in Years ($\\tau$)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    # ---------------------------------------------------------
    # Sweep 3: Varying Volatility (sigma)
    # ---------------------------------------------------------
    sigma_sweep = np.linspace(0.05, 0.8, N).astype(np.float32)
    
    inputs_3 = torch.tensor(np.column_stack([
        np.full(N, M_fixed), 
        np.full(N, tau_fixed), 
        np.full(N, r_fixed), 
        np.full(N, q_fixed), 
        sigma_sweep
    ]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        v_hat_3 = model.nn2(inputs_3).cpu().numpy().flatten()
        
    v_true_3 = analytical_black_scholes(M_fixed, tau_fixed, r_fixed, q_fixed, sigma_sweep)
    
    axes[2].plot(sigma_sweep, v_true_3, 'k-', lw=2, label='Analytical BS')
    axes[2].scatter(sigma_sweep, v_hat_3, c='red', s=15, label='PINN_MLP Predict', zorder=3)
    axes[2].set_title(f'Varying Volatility ($\\sigma$)\nM={M_fixed}, $\\tau$={tau_fixed}, r={r_fixed}')
    axes[2].set_xlabel('Volatility ($\\sigma$)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('analytical_sweeps.svg')
    plt.show()
    print("Saved analytical sweeps plot as 'analytical_sweeps.svg'")

def plot_test_dataset(model, test_loader, device):
    print("Evaluating Test Dataset...")
    model.eval()
    
    v_true_list = []
    v_hat_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
            _, v_hat = model(M, tau, u, r, q)
            
            v_true_list.append(v_true.cpu().numpy())
            v_hat_list.append(v_hat.cpu().numpy())
            
    v_true_all = np.concatenate(v_true_list).flatten()
    v_hat_all = np.concatenate(v_hat_list).flatten()
    
    mse = np.mean((v_true_all - v_hat_all)**2)
    
    # Calculate Percentage Error
    # Added epsilon (1e-6) to denominator to avoid division by zero for OTM options
    pct_error = ((v_hat_all - v_true_all) / np.maximum(v_true_all, 1e-6)) * 100
    
    # Create 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Plot 1: Scatter Predictions vs True ---
    axes[0].scatter(v_true_all, v_hat_all, alpha=0.3, s=5, c='blue', label='Predictions')
    min_val = min(v_true_all.min(), v_hat_all.min())
    max_val = max(v_true_all.max(), v_hat_all.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
    
    axes[0].set_title(f'Test Dataset: Predicted vs True Normalized Price\nMSE: {mse:.6f}')
    axes[0].set_xlabel('True Normalized Price ($v_{true}$)')
    axes[0].set_ylabel('Predicted Normalized Price ($v_{hat}$)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)
    
    # --- Plot 2: Percentage Error ---
    axes[1].scatter(v_true_all, pct_error, alpha=0.3, s=5, c='orange')
    axes[1].axhline(0, color='red', linestyle='--', label='Zero Error')
    
    axes[1].set_title('Percentage Error vs. True Price')
    axes[1].set_xlabel('True Normalized Price ($v_{true}$)')
    axes[1].set_ylabel('Percentage Error (%)')
    # Cap the Y-axis to avoid zero-division explosions from deeply OTM options ruining the plot
    axes[1].set_ylim(-50, 50) 
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('test_predictions_scatter.png')
    plt.show()
    print("Saved test scatter and error plot as 'test_predictions_scatter.png'")


def plot_pde_consistency(model, test_loader, device):
    print("Testing Network 2's PDE Consistency...")
    model.eval()
    
    v_hat_list = []
    v_analytical_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            M, tau, u, r, q, _, _ = [b.to(device) for b in batch]
            
            # Forward pass
            sigma_hat, v_hat = model(M, tau, u, r, q)
            
            M_np = M.cpu().numpy().flatten()
            tau_np = tau.cpu().numpy().flatten()
            r_np = r.cpu().numpy().flatten()
            q_np = q.cpu().numpy().flatten()
            sigma_hat_np = sigma_hat.cpu().numpy().flatten()
            
            # Analytical Baseline using network-generated volatility
            v_analytical = analytical_black_scholes(M_np, tau_np, r_np, q_np, sigma_hat_np)
            
            v_hat_list.append(v_hat.cpu().numpy().flatten())
            v_analytical_list.append(v_analytical)
            
    v_hat_all = np.concatenate(v_hat_list)
    v_analytical_all = np.concatenate(v_analytical_list)
    
    mse_pde = np.mean((v_hat_all - v_analytical_all)**2)
    
    # Calculate Percentage Error
    pct_error = ((v_hat_all - v_analytical_all) / np.maximum(v_analytical_all, 1e-6)) * 100
    
    # Create 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Plot 1: Network 2 vs Analytical Formula ---
    axes[0].scatter(v_analytical_all, v_hat_all, alpha=0.3, s=5, c='green', label='Network 2 Predictions')
    min_val = min(v_analytical_all.min(), v_hat_all.min())
    max_val = max(v_analytical_all.max(), v_hat_all.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect PDE Match (y=x)')
    
    axes[0].set_title(f'PDE Consistency: Network 2 vs Analytical BS\nMSE: {mse_pde:.6f}')
    axes[0].set_xlabel('Analytical Price (using Network 1 Volatility)')
    axes[0].set_ylabel('Network 2 Predicted Price ($v_{hat}$)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)
    
    # --- Plot 2: Percentage Error ---
    axes[1].scatter(v_analytical_all, pct_error, alpha=0.3, s=5, c='purple')
    axes[1].axhline(0, color='red', linestyle='--', label='Zero Error')
    
    axes[1].set_title('PDE Match Percentage Error vs. Analytical Price')
    axes[1].set_xlabel('Analytical Price (using Network 1 Volatility)')
    axes[1].set_ylabel('Percentage Error (%)')
    # Cap the Y-axis to avoid zero-division explosions from deeply OTM options ruining the plot
    axes[1].set_ylim(-50, 50) 
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('pde_consistency_scatter.png')
    plt.show()
    print("Saved PDE consistency scatter and error plot as 'pde_consistency_scatter.png'")




def compute_network_vega(model, M, tau, u, r, q):
    """
    Computes the Vega (dv_hat / d_sigma) using PyTorch Automatic Differentiation.
    """
    model.eval()
    
    # 1. Forward pass
    sigma_hat, v_hat = model(M, tau, u, r, q)
    
    # 2. Compute the gradient of v_hat with respect to sigma_hat
    # We use retain_graph=True in case you want to compute other Greeks afterward
    vega_hat = torch.autograd.grad(
        outputs=v_hat, 
        inputs=sigma_hat, 
        grad_outputs=torch.ones_like(v_hat),
        create_graph=False, 
        retain_graph=True
    )[0]
    
    return vega_hat, sigma_hat

def analytical_normalized_vega(M, tau, r, q, sigma):
    """
    Calculates the analytical Black-Scholes Vega for the normalized price (v = C/K).
    Standard Vega: S * sqrt(tau) * e^(-q*tau) * N'(d1)
    Normalized Vega: M * sqrt(tau) * e^(-q*tau) * N'(d1)
    """
    # Prevent division by zero
    tau = np.maximum(tau, 1e-8)
    
    d1 = (np.log(M) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    
    # norm.pdf is the derivative of the CDF ( N'(d1) )
    vega = M * np.sqrt(tau) * np.exp(-q * tau) * norm.pdf(d1)
    
    # If maturity is zero, Vega is zero
    vega = np.where(tau <= 1e-8, 0.0, vega)
    return vega

def plot_vega_consistency(model, test_loader, device):
    print("Testing Network Vega vs Analytical Vega...")
    model.eval()
    
    vega_network_list = []
    vega_analytical_list = []
    
    for batch in test_loader:
        M, tau, u, r, q, _, _ = [b.to(device) for b in batch]
        
        # We need M, tau, r, q to require grad? 
        # Actually, to compute Vega, we ONLY need sigma_hat to exist in the computation graph.
        # But autograd requires the output to trace back to the input. 
        # Since sigma_hat is generated by Network 1, it is part of the graph.
        
        vega_net_tensor, sigma_hat_tensor = compute_network_vega(model, M, tau, u, r, q)
        
        # Convert to numpy for the analytical function
        M_np = M.detach().cpu().numpy().flatten()
        tau_np = tau.detach().cpu().numpy().flatten()
        r_np = r.detach().cpu().numpy().flatten()
        q_np = q.detach().cpu().numpy().flatten()
        sigma_np = sigma_hat_tensor.detach().cpu().numpy().flatten()
        vega_net_np = vega_net_tensor.detach().cpu().numpy().flatten()
        
        # Calculate analytical Vega using Network 1's latent sigma
        vega_analytical = analytical_normalized_vega(M_np, tau_np, r_np, q_np, sigma_np)
        
        vega_network_list.append(vega_net_np)
        vega_analytical_list.append(vega_analytical)
            
    vega_net_all = np.concatenate(vega_network_list)
    vega_analytical_all = np.concatenate(vega_analytical_list)
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(vega_analytical_all, vega_net_all, alpha=0.3, s=5, c='purple', label='Network Vega (Autograd)')
    
    min_val = min(vega_analytical_all.min(), vega_net_all.min())
    max_val = max(vega_analytical_all.max(), vega_net_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match (y=x)')
    
    plt.title('Vega Consistency: Autograd vs Analytical')
    plt.xlabel('Analytical Vega (using Network 1 $\\sigma$)')
    plt.ylabel('Network Autograd Vega ($\\partial v / \\partial \\sigma$)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ==========================================
# EXPERIMENT 1: GLOBAL METRICS & RESIDUALS
# ==========================================
def experiment_1_global_metrics(model, test_loader, device):
    print("\n--- EXPERIMENT 1: Global Error Metrics ---")
    model.eval()
    
    v_true_list = []
    v_hat_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
            _, v_hat = model(M, tau, u, r, q)
            v_true_list.append(v_true.cpu().numpy().flatten())
            v_hat_list.append(v_hat.cpu().numpy().flatten())
            
    v_true = np.concatenate(v_true_list)
    v_hat = np.concatenate(v_hat_list)
    
    # Calculate Metrics
    mse = np.mean((v_true - v_hat)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(v_true - v_hat))
    
    # MAPE (Capped to avoid division by zero on Deep OTM options)
    epsilon = 1e-6
    mape = np.mean(np.abs(v_true - v_hat) / np.maximum(v_true, epsilon)) * 100
    
    # R^2 Score
    ss_res = np.sum((v_true - v_hat)**2)
    ss_tot = np.sum((v_true - np.mean(v_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"MSE:   {mse:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"MAE:   {mae:.6f}")
    print(f"MAPE:  {mape:.2f}%")
    print(f"R^2:   {r2:.6f}")
    
    # Plot Residual Distribution
    residuals = v_hat - v_true
    plt.figure(figsize=(8, 5))
    
    # MODIFIED: Added range=(-0.25, 0.025) to restrict bin calculation
    plt.hist(residuals, bins=100, range=(-0.025, 0.025), alpha=0.7, color='teal', edgecolor='black')
    
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.title(f'Distribution of Residuals (Predicted - True)\nMean Error: {np.mean(residuals):.6f}')
    plt.xlabel('Error Value (Normalized Price)')
    plt.ylabel('Frequency')
    
    # MODIFIED: Locked the x-axis display limits
    plt.xlim(-0.1, 0.1)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp1_residual_histogram.png')
    plt.show()


# ==========================================
# EXPERIMENT 2: ERROR STRATIFICATION HEATMAP
# ==========================================
def experiment_2_error_heatmap(model, test_loader, device):
    print("\n--- EXPERIMENT 2: Error Stratification Heatmap ---")
    model.eval()
    
    M_list, tau_list, error_list = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
            _, v_hat = model(M, tau, u, r, q)
            
            M_list.append(M.cpu().numpy().flatten())
            tau_list.append(tau.cpu().numpy().flatten())
            
            # Use Absolute Error (MAE) for the heatmap
            abs_err = np.abs(v_true.cpu().numpy().flatten() - v_hat.cpu().numpy().flatten())
            error_list.append(abs_err)
            
    M_all = np.concatenate(M_list)
    tau_all = np.concatenate(tau_list)
    error_all = np.concatenate(error_list)
    
    # Define Bins for Moneyness (M) and Maturity (tau)
    # M < 0.95 (Deep OTM), 0.95-0.98 (OTM), 0.98-1.02 (ATM), 1.02-1.05 (ITM), > 1.05 (Deep ITM)
    M_bins = [0.0, 0.95, 0.98, 1.02, 1.05, 3.0]
    M_labels = ['Deep OTM', 'OTM', 'ATM', 'ITM', 'Deep ITM']
    
    # tau < 0.1 (Short), 0.1-0.5 (Medium), > 0.5 (Long)
    tau_bins = [0.0, 0.1, 0.5, 6.0]
    tau_labels = ['Short (<1m)', 'Med (1-6m)', 'Long (>6m)']
    
    # Compute 2D binned statistics (Mean Absolute Error per bin)
    heatmap_data, _, _, _ = binned_statistic_2d(
        x=tau_all, y=M_all, values=error_all, statistic='mean', bins=[tau_bins, M_bins]
    )
    
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap_data, origin='lower', aspect='auto', cmap='Reds')
    plt.colorbar(label='Mean Absolute Error')
    
    # Set ticks and labels
    plt.xticks(ticks=np.arange(len(M_labels)), labels=M_labels)
    plt.yticks(ticks=np.arange(len(tau_labels)), labels=tau_labels)
    
    # Annotate text in cells
    for i in range(len(tau_labels)):
        for j in range(len(M_labels)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                plt.text(j, i, f'{val:.4f}', ha='center', va='center', 
                         color='white' if val > np.nanmax(heatmap_data)/2 else 'black')
                
    plt.title('Pricing Error (MAE) Stratified by Moneyness and Maturity')
    plt.xlabel('Moneyness ($M = S/K$)')
    plt.ylabel('Time to Maturity ($\\tau$)')
    plt.tight_layout()
    plt.savefig('exp2_error_heatmap.png')
    plt.show()


# ==========================================
# EXPERIMENT 3: GREEKS (DELTA & GAMMA)
# ==========================================
def analytical_greeks(M, tau, r, q, sigma):
    """Computes Analytical Delta and Normalized Gamma."""
    tau = np.maximum(tau, 1e-8)
    d1 = (np.log(M) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    
    delta = np.exp(-q * tau) * norm.cdf(d1)
    
    # Normalized Gamma: K * Gamma = N'(d1) * e^(-q*tau) / (M * sigma * sqrt(tau))
    gamma_norm = (np.exp(-q * tau) * norm.pdf(d1)) / (M * sigma * np.sqrt(tau))
    gamma_norm = np.where(tau <= 1e-8, 0.0, gamma_norm)
    
    return delta, gamma_norm

def experiment_3_greeks(model, test_loader, device):
    print("\n--- EXPERIMENT 3: Neural Greeks vs Analytical Greeks ---")
    model.eval()
    
    delta_net_list, gamma_net_list = [], []
    delta_ana_list, gamma_ana_list = [], []
    
    for batch in test_loader:
        M, tau, u, r, q, _, _ = [b.to(device) for b in batch]
        
        # Enable gradient tracking for M
        M = M.clone().requires_grad_(True)
        
        # Forward pass
        sigma_hat, v_hat = model(M, tau, u, r, q)
        
        # Network Delta (dv/dM)
        dv_dM = torch.autograd.grad(
            outputs=v_hat, inputs=M, 
            grad_outputs=torch.ones_like(v_hat),
            create_graph=True, retain_graph=True
        )[0]
        
        # Network Normalized Gamma (d2v/dM2)
        d2v_dM2 = torch.autograd.grad(
            outputs=dv_dM, inputs=M, 
            grad_outputs=torch.ones_like(dv_dM),
            create_graph=False, retain_graph=False
        )[0]
        
        # Detach and convert to NumPy
        M_np = M.detach().cpu().numpy().flatten()
        tau_np = tau.detach().cpu().numpy().flatten()
        r_np = r.detach().cpu().numpy().flatten()
        q_np = q.detach().cpu().numpy().flatten()
        sigma_np = sigma_hat.detach().cpu().numpy().flatten()
        
        delta_net_list.append(dv_dM.detach().cpu().numpy().flatten())
        gamma_net_list.append(d2v_dM2.detach().cpu().numpy().flatten())
        
        # Compute Analytical Baseline using the Network's latent volatility
        delta_ana, gamma_ana = analytical_greeks(M_np, tau_np, r_np, q_np, sigma_np)
        delta_ana_list.append(delta_ana)
        gamma_ana_list.append(gamma_ana)
        
    # Concatenate results
    delta_net = np.concatenate(delta_net_list)
    gamma_net = np.concatenate(gamma_net_list)
    delta_ana = np.concatenate(delta_ana_list)
    gamma_ana = np.concatenate(gamma_ana_list)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Delta Plot
    axes[0].scatter(delta_ana, delta_net, alpha=0.3, s=5, c='purple')
    min_val, max_val = min(delta_ana.min(), delta_net.min()), max(delta_ana.max(), delta_net.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match')
    axes[0].set_title('Delta ($\\Delta$): Network Autograd vs Analytical')
    axes[0].set_xlabel('Analytical $\\Delta$ (using Network 1 $\\sigma$)')
    axes[0].set_ylabel('Network Autograd $\\Delta$ ($\\partial v / \\partial M$)')
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()
    
    # Gamma Plot
    axes[1].scatter(gamma_ana, gamma_net, alpha=0.3, s=5, c='orange')
    min_val, max_val = 0, np.percentile(gamma_ana, 99) # Cap outliers for clean visualization
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match')
    axes[1].set_xlim(min_val, max_val)
    axes[1].set_ylim(min_val, max_val)
    axes[1].set_title('Normalized Gamma ($\\Gamma$): Network Autograd vs Analytical')
    axes[1].set_xlabel('Analytical Normalized $\\Gamma$')
    axes[1].set_ylabel('Network Autograd Normalized $\\Gamma$ ($\\partial^2 v / \\partial M^2$)')
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('exp3_greeks_scatter.png')
    plt.show()


# ==========================================
# EXPERIMENT 4: VOLATILITY REGIMES (FNO IMPACT)
# ==========================================
# def experiment_4_volatility_regimes(model, test_loader, device):
#     print("\n--- EXPERIMENT 4: FNO Branch Impact (Volatility Regimes) ---")
#     model.eval()
    
#     u_var_list = []
#     error_list = []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
#             _, v_hat = model(M, tau, u, r, q)
            
#             # Compute the variance of the input grid 'u' as a proxy for the market's volatility regime
#             # Flatten the non-batch dimensions to compute variance per sample
#             u_flat = u.view(u.size(0), -1)
#             u_var = torch.var(u_flat, dim=1)
            
#             u_var_list.append(u_var.cpu().numpy())
            
#             # Use Absolute Error
#             abs_err = torch.abs(v_true - v_hat)
#             error_list.append(abs_err.cpu().numpy().flatten())
            
#     u_var_all = np.concatenate(u_var_list)
#     error_all = np.concatenate(error_list)
    
#     # Determine the quartile thresholds for the volatility regimes
#     q25, q50, q75 = np.percentile(u_var_all, [25, 50, 75])
    
#     # Group the errors into the 4 regimes
#     regime_1 = error_all[u_var_all <= q25]
#     regime_2 = error_all[(u_var_all > q25) & (u_var_all <= q50)]
#     regime_3 = error_all[(u_var_all > q50) & (u_var_all <= q75)]
#     regime_4 = error_all[u_var_all > q75]
    
#     data_to_plot = [regime_1, regime_2, regime_3, regime_4]
#     labels = ['Low Vol\n(Bottom 25%)', 'Med Vol\n(25-50%)', 'High Vol\n(50-75%)', 'Extreme Vol\n(Top 25%)']
    
#     plt.figure(figsize=(9, 6))
    
#     # Create the boxplot. We set showfliers=False to hide extreme outliers 
#     # so the main "box" of the distribution is visually readable.
#     box = plt.boxplot(data_to_plot, patch_artist=True, labels=labels, showfliers=False)
    
#     # Color the boxes using a sequential blue colormap
#     colors = ['#add8e6', '#87cefa', '#4682b4', '#000080']
#     for patch, color in zip(box['boxes'], colors):
#         patch.set_facecolor(color)
        
#     plt.title('Pricing Error (MAE) Across Market Volatility Regimes')
#     plt.ylabel('Absolute Error (Normalized Price)')
#     plt.grid(True, alpha=0.3, axis='y')
    
#     # Calculate and annotate the mean error above each box
#     means = [np.mean(r) for r in data_to_plot]
#     for i, mean_val in enumerate(means):
#         # Position the text slightly above the top whisker
#         top_whisker = box['caps'][i*2+1].get_ydata()[0]
#         plt.text(i + 1, top_whisker + np.max(error_all)*0.01, f'Mean: {mean_val:.4f}', 
#                  ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
                 
#     plt.tight_layout()
#     plt.savefig('exp4_volatility_regimes.png')
#     plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# ==========================================
# EXPERIMENT 4: VOLATILITY REGIMES (FIXED)
# ==========================================
def experiment_4_volatility_regimes(model, test_loader, device):
    print("\n--- EXPERIMENT 4: Error Across Volatility Regimes ---")
    model.eval()
    
    error_list = []
    u_mean_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
            _, v_hat = model(M, tau, u, r, q)
            
            # Calculate Absolute Error
            abs_err = torch.abs(v_true - v_hat).cpu().numpy().flatten()
            error_list.extend(abs_err)
            
            # We average across the 374 features (dim=1) to get the mean surface IV.
            u_mean = torch.mean(u, dim=1).cpu().numpy().flatten()
            u_mean_list.extend(u_mean)
            
    df = pd.DataFrame({'MAE': error_list, 'Surface_Mean_IV': u_mean_list})
    
    # Sort into quartiles based on the average height of the IV surface
    # Added duplicates='drop' to handle potential days with identical average IVs safely
    df['Regime'] = pd.qcut(df['Surface_Mean_IV'], q=4, labels=[
        'Low Vol\n(Bottom 25%)', 
        'Med Vol\n(25-50%)', 
        'High Vol\n(50-75%)', 
        'Extreme Vol\n(Top 25%)'
    ], duplicates='drop')
    
    # Calculate means for the annotations (observed=True handles pandas future warnings)
    means = df.groupby('Regime', observed=True)['MAE'].mean().values
    
    # Plotting
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Regime', y='MAE', data=df, palette='Blues', showfliers=False, width=0.4)
    
    # Dynamically grab the top of the Y-axis so our text annotations stay visible
    y_max = ax.get_ylim()[1]
    
    # Add mean text above the boxes
    for i, mean_val in enumerate(means):
        ax.text(i, y_max * 0.95, f'Mean: {mean_val:.4f}', 
                ha='center', va='top', color='maroon', fontweight='bold')
                
    plt.title('Pricing Error (MAE) Across Market Volatility Regimes')
    plt.ylabel('Absolute Error (Normalized Price)')
    plt.xlabel('')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp4_volatility_regimes.png')
    plt.show()
    print("Saved experiment 4 plot to 'exp4_volatility_regimes.png'")

# ==========================================
# EXPERIMENT 5: COMPUTATIONAL EFFICIENCY
# ==========================================
def experiment_5_benchmark(model, test_loader, device):
    print("\n--- EXPERIMENT 5: Inference Speed Benchmark ---")
    model.eval()
    
    # 1. Benchmark PyTorch PINN
    total_options = 0
    pinn_start_time = time.time()
    
    # We pre-load batches to isolate pure network inference time (excluding data IO)
    batches = [[b.to(device) for b in batch] for batch in test_loader]
    
    with torch.no_grad():
        for batch in batches:
            M, tau, u, r, q, _, _ = batch
            total_options += M.size(0)
            _ = model(M, tau, u, r, q)
            
    # Torch sync to ensure GPU operations are finished
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    pinn_end_time = time.time()
    pinn_duration = pinn_end_time - pinn_start_time
    pinn_throughput = total_options / pinn_duration
    
    print(f"Hardware: {device}")
    print(f"Total Options Priced: {total_options:,}")
    print(f"PINN Total Time:      {pinn_duration:.4f} seconds")
    print(f"PINN Throughput:      {pinn_throughput:,.0f} options / second")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize model architecture (must match training parameters)
    model = DeepBS_Solver(hidden_dim=64, num_layers=4, p_dim=64).to(device)
    
    # 2. Load the trained weights
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("Successfully loaded 'best_model.pth'")
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Please ensure the path is correct.")
        return

    # 3. Load the test dataset for the scatter plot
    print("Loading test dataset...")
    test_ds = ChronologicalOptionDataset('../wrds_data/deeponet_tensors.h5', 'test')
    test_loader = DataLoader(
        test_ds, 
        batch_size=512, 
        shuffle=False,
        collate_fn=create_pinn_collate_fn(colloc_ratio=0.0)
    )
    
    # 4. Execute the plots
    # plot_test_dataset(model, test_loader, device)
    # # plot_analytical_sweeps(model, device)
    # plot_pde_consistency(model, test_loader, device)
    # # plot_vega_consistency(model, test_loader, device)
    
    # New Experiments
    experiment_1_global_metrics(model, test_loader, device)
    experiment_2_error_heatmap(model, test_loader, device)
    experiment_3_greeks(model, test_loader, device)
    experiment_4_volatility_regimes(model, test_loader, device)
    experiment_5_benchmark(model, test_loader, device)

if __name__ == "__main__":
    main()