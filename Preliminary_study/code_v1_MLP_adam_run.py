"""
Deep Black-Scholes Solver with Physics-Informed Neural Networks
Complete corrected implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt  # <-- Added for plotting

# ==========================================
# NETWORK ARCHITECTURE
# ==========================================

class PINN_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, is_positive=True):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        if is_positive:
            layers.append(nn.Softplus())
            
        self.net = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.net(x)


class DeepBS_Solver(nn.Module):
    def __init__(self, u_dim=374, hidden_dim=128, num_layers=4):
        super().__init__()
        
        self.nn1 = PINN_MLP(
            input_dim=2 + u_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=1, 
            is_positive=True
        )
        
        self.nn2 = PINN_MLP(
            input_dim=5, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=1, 
            is_positive=True
        )

    def forward(self, M, tau, u, r, q):
        nn1_input = torch.cat([M, tau, u], dim=1)
        sigma = self.nn1(nn1_input)
        
        nn2_input = torch.cat([M, tau, r, q, sigma], dim=1)
        v_hat = self.nn2(nn2_input)
        
        return sigma, v_hat


# ==========================================
# LOSS FUNCTIONS
# ==========================================

def compute_derivatives(v_hat, M, tau):
    dv_dM = torch.autograd.grad(
        outputs=v_hat, inputs=M, 
        grad_outputs=torch.ones_like(v_hat),
        create_graph=True, retain_graph=True
    )[0]
    
    d2v_dM2 = torch.autograd.grad(
        outputs=dv_dM, inputs=M, 
        grad_outputs=torch.ones_like(dv_dM),
        create_graph=True, retain_graph=True
    )[0]
    
    dv_dtau = torch.autograd.grad(
        outputs=v_hat, inputs=tau, 
        grad_outputs=torch.ones_like(v_hat),
        create_graph=True, retain_graph=True
    )[0]
    
    return dv_dM, d2v_dM2, dv_dtau


def generate_boundary_points(batch_size, u, r, q, device, M_max=3.0, tau_max=5.32):
    indices = torch.randint(0, u.size(0), (batch_size,), device=device)

    # Terminal condition at tau = 0, M ∈ [0, M_max]
    M_term = torch.rand(batch_size, 1, device=device) * M_max
    tau_term = torch.zeros(batch_size, 1, device=device)

    # Lower boundary at M = 0, tau ∈ [0, tau_max]
    M_lower = torch.zeros(batch_size, 1, device=device)
    tau_lower = torch.rand(batch_size, 1, device=device) * tau_max

    # Upper boundary at M = M_max, tau ∈ [0, tau_max]
    M_upper = torch.full((batch_size, 1), M_max, device=device)
    tau_upper = torch.rand(batch_size, 1, device=device) * tau_max

    return {
        'terminal': (M_term, tau_term, u[indices], r[indices], q[indices]),
        'lower': (M_lower, tau_lower, u[indices], r[indices], q[indices]),
        'upper': (M_upper, tau_upper, u[indices], r[indices], q[indices]),
    }


def pinn_loss_fn(
    model, M, tau, u, r, q, v_true, mask, lambdas,
    device, num_boundary=256,
    M_max=3.0, tau_max=5.32
):
    sigma_hat, v_hat = model(M, tau, u, r, q)

    # ----------------
    # Data Loss
    # ----------------
    squared_errors = (v_hat - v_true) ** 2
    loss_data = torch.sum(squared_errors * mask) / (torch.sum(mask) + 1e-8)

    # ----------------
    # PDE Loss (tau is in YEARS; no scaling factor needed for time derivative)
    # ----------------
    dv_dM, d2v_dM2, dv_dtau = compute_derivatives(v_hat, M, tau)

    pde_residual = (
        -dv_dtau
        + 0.5 * (sigma_hat ** 2) * (M ** 2) * d2v_dM2
        + (r - q) * M * dv_dM
        - r * v_hat
    )
    loss_pde = torch.mean(pde_residual ** 2)

    # ----------------
    # Arbitrage Loss
    # ----------------
    loss_arb = torch.mean(F.relu(-dv_dM)) + torch.mean(F.relu(-d2v_dM2))

    # ----------------
    # Boundary Conditions (tau in YEARS)
    # ----------------
    boundaries = generate_boundary_points(
        num_boundary, u, r, q, device, M_max=M_max, tau_max=tau_max
    )

    # Terminal: v(M,0) = max(M-1,0)
    M_t, tau_t, u_t, r_t, q_t = boundaries['terminal']
    M_t.requires_grad_(True)
    tau_t.requires_grad_(True)
    _, v_term = model(M_t, tau_t, u_t, r_t, q_t)
    loss_term = F.mse_loss(v_term, F.relu(M_t - 1.0))

    # Lower: v(0,tau)=0
    M_l, tau_l, u_l, r_l, q_l = boundaries['lower']
    M_l.requires_grad_(True)
    tau_l.requires_grad_(True)
    _, v_lower = model(M_l, tau_l, u_l, r_l, q_l)
    loss_lower = F.mse_loss(v_lower, torch.zeros_like(v_lower))

    # Upper: v(M_max,tau) = M_max*e^{-q tau} - e^{-r tau}   (continuous compounding, tau in years)
    M_u, tau_u, u_u, r_u, q_u = boundaries['upper']
    M_u.requires_grad_(True)
    tau_u.requires_grad_(True)
    _, v_upper = model(M_u, tau_u, u_u, r_u, q_u)
    v_upper_true = M_u * torch.exp(-q_u * tau_u) - torch.exp(-r_u * tau_u)
    loss_upper = F.mse_loss(v_upper, v_upper_true)

    # ----------------
    # Total
    # ----------------
    loss_total = (
        lambdas.get('data', 1.0) * loss_data +
        lambdas.get('pde', 0.1) * loss_pde +
        lambdas.get('arb', 0.1) * loss_arb +
        lambdas.get('term', 1.0) * loss_term +
        lambdas.get('lower', 1.0) * loss_lower +
        lambdas.get('upper', 1.0) * loss_upper
    )

    loss_dict = {
        'data': loss_data.item(),
        'pde': loss_pde.item(),
        'arb': loss_arb.item(),
        'term': loss_term.item(),
        'lower': loss_lower.item(),
        'upper': loss_upper.item(),
        'total': loss_total.item()
    }

    return loss_total, loss_dict


# ==========================================
# DATASET
# ==========================================

class ChronologicalOptionDataset(Dataset):
    def __init__(self, h5_path, split='train', load_to_memory=True):
        self.h5_path = h5_path
        self.load_to_memory = load_to_memory
        
        with h5py.File(h5_path, 'r') as f:
            total_rows = f['target_v'].shape[0]
            self.u_dim = f['branch_u'].shape[1]
            
            train_end = int(0.80 * total_rows)
            val_end = int(0.90 * total_rows)
            
            if split == 'train':
                self.start_idx, self.end_idx = 0, train_end
            elif split == 'val':
                self.start_idx, self.end_idx = train_end, val_end
            elif split == 'test':
                self.start_idx, self.end_idx = val_end, total_rows
            else:
                raise ValueError("Split must be 'train', 'val', or 'test'.")
            
            self.length = self.end_idx - self.start_idx
            
            if load_to_memory:
                self.branch_u = f['branch_u'][self.start_idx:self.end_idx]
                self.trunk_y = f['trunk_y'][self.start_idx:self.end_idx]
                self.target_v = f['target_v_normalized'][self.start_idx:self.end_idx]
                
        print(f"Loaded {split}: {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.branch_u[idx], dtype=torch.float32),
            torch.tensor(self.trunk_y[idx], dtype=torch.float32),
            torch.tensor(self.target_v[idx], dtype=torch.float32)
        )
    
    def get_u_dim(self):
        return self.u_dim


# ==========================================
# COLLATE FUNCTION
# ==========================================

# def create_pinn_collate_fn(colloc_ratio=0.2, M_max=3.0, tau_max=5.32):
#     def collate_fn(batch):
#         u_true = torch.stack([item[0] for item in batch])
#         y_true = torch.stack([item[1] for item in batch])
#         v_true = torch.stack([item[2] for item in batch])

#         if v_true.dim() == 1:
#             v_true = v_true.unsqueeze(1)

#         M_true = y_true[:, 0:1]
#         tau_true = y_true[:, 1:2]   # already in years
#         r_true = y_true[:, 2:3]
#         q_true = y_true[:, 3:4]

#         num_true = len(batch)

#         if colloc_ratio > 0.0:
#             num_colloc = max(1, int(num_true * (colloc_ratio / (1.0 - colloc_ratio))))

#             # Collocation domain in (M, tau_years)
#             M_colloc = torch.rand(num_colloc, 1) * M_max
#             tau_colloc = torch.rand(num_colloc, 1) * tau_max

#             idx = torch.randint(0, num_true, (num_colloc,))
#             u_colloc = u_true[idx]
#             r_colloc = r_true[idx]
#             q_colloc = q_true[idx]
#             v_colloc = torch.zeros(num_colloc, 1)

#             M_final = torch.cat([M_true, M_colloc])
#             tau_final = torch.cat([tau_true, tau_colloc])
#             u_final = torch.cat([u_true, u_colloc])
#             r_final = torch.cat([r_true, r_colloc])
#             q_final = torch.cat([q_true, q_colloc])
#             v_final = torch.cat([v_true, v_colloc])
#             mask = torch.cat([torch.ones(num_true, 1), torch.zeros(num_colloc, 1)])
#         else:
#             M_final, tau_final, u_final = M_true, tau_true, u_true
#             r_final, q_final, v_final = r_true, q_true, v_true
#             mask = torch.ones(num_true, 1)

#         return M_final, tau_final, u_final, r_final, q_final, v_final, mask

#     return collate_fn


def create_pinn_collate_fn(colloc_ratio=0.2, M_max=3.0, tau_max=5.32, points_per_u=10):
    def collate_fn(batch):
        u_true = torch.stack([item[0] for item in batch])
        y_true = torch.stack([item[1] for item in batch])
        v_true = torch.stack([item[2] for item in batch])

        if v_true.dim() == 1:
            v_true = v_true.unsqueeze(1)

        M_true = y_true[:, 0:1]
        tau_true = y_true[:, 1:2]
        r_true = y_true[:, 2:3]
        q_true = y_true[:, 3:4]

        num_true = len(batch)

        if colloc_ratio > 0.0:
            target_num_colloc = max(1, int(num_true * (colloc_ratio / (1.0 - colloc_ratio))))
            num_unique_u = max(1, target_num_colloc // points_per_u)
            actual_num_colloc = num_unique_u * points_per_u

            idx = torch.randint(0, num_true, (num_unique_u,))
            u_base = u_true[idx]
            r_base = r_true[idx]
            q_base = q_true[idx]
            
            u_colloc = u_base.repeat_interleave(points_per_u, dim=0)
            r_colloc = r_base.repeat_interleave(points_per_u, dim=0)
            q_colloc = q_base.repeat_interleave(points_per_u, dim=0)
            
            M_colloc = torch.rand(actual_num_colloc, 1) * M_max
            tau_colloc = torch.rand(actual_num_colloc, 1) * tau_max
            v_colloc = torch.zeros(actual_num_colloc, 1)

            M_final = torch.cat([M_true, M_colloc])
            tau_final = torch.cat([tau_true, tau_colloc])
            u_final = torch.cat([u_true, u_colloc])
            r_final = torch.cat([r_true, r_colloc])
            q_final = torch.cat([q_true, q_colloc])
            v_final = torch.cat([v_true, v_colloc])
            mask = torch.cat([torch.ones(num_true, 1), torch.zeros(actual_num_colloc, 1)])
            
        else:
            M_final, tau_final, u_final = M_true, tau_true, u_true
            r_final, q_final, v_final = r_true, q_true, v_true
            mask = torch.ones(num_true, 1)

        return M_final, tau_final, u_final, r_final, q_final, v_final, mask

    return collate_fn


# ==========================================
# TRAINING
# ==========================================

def train_model():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = {
        'epochs_adam': 100,      
        'epochs_lbfgs': 50,
        'batch_size': 256,      #changed
        'lr': 1e-3,
        'hidden_dim': 64,
        'num_layers': 4, 
        'colloc_ratio': 0.2,
        'points_per_u': 20,      #changed
        'num_boundary': 128, 
        'grad_clip': 1.0,
        'M_max': 3.0,
        'tau_max': 5.32,
        'patience_adam': 3,     #changed
        'patience_lbfgs': 3,    #changed
    }
    
    lambdas = {'data': 1.0, 'pde': 0.5, 'arb': 0.5, 'term': 0.5, 'lower': 0.5, 'upper': 0.5}
    
    # Data
    print("Loading datasets...")
    train_ds = ChronologicalOptionDataset('../wrds_data/deeponet_tensors.h5', 'train')
    val_ds = ChronologicalOptionDataset('../wrds_data/deeponet_tensors.h5', 'val')
    test_ds = ChronologicalOptionDataset('../wrds_data/deeponet_tensors.h5', 'test') 
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    
    train_loader = DataLoader(
        train_ds,
        config['batch_size'],
        shuffle=True,
        collate_fn=create_pinn_collate_fn(
            colloc_ratio=config['colloc_ratio'],
            M_max=config['M_max'],
            tau_max=config['tau_max'],
            points_per_u=config['points_per_u'],
        )
    )
    
    val_loader = DataLoader(
        val_ds,
        config['batch_size'] * 2,
        shuffle=False,
        collate_fn=create_pinn_collate_fn(
            colloc_ratio=0.0,
            M_max=config['M_max'],
            tau_max=config['tau_max'],
        )
    )
    
    test_loader = DataLoader(
        test_ds,
        config['batch_size'] * 2,
        shuffle=False,
        collate_fn=create_pinn_collate_fn(
            colloc_ratio=0.0,
            M_max=config['M_max'],
            tau_max=config['tau_max'],
        )
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = DeepBS_Solver(train_ds.get_u_dim(), config['hidden_dim'], config['num_layers']).to(device)
    print(f"Model initialized with u_dim={train_ds.get_u_dim()}, hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")
    
    # Optimizers
    optimizer_adam = optim.Adam(model.parameters(), lr=config['lr'])
    #changed
    scheduler = ReduceLROnPlateau(optimizer_adam, 'min', factor=0.5, patience=2)
    
    best_val = float('inf')
    
    # --- Lists to track history for plotting and saving ---
    all_train_losses = []
    all_val_losses = []
    
    # ==========================================
    # HELPER: EVALUATION LOOP (Generalized)
    # ==========================================
    def evaluate_model(loader, dataset_len):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
                _, v_hat = model(M, tau, u, r, q)
                total_loss += F.mse_loss(v_hat, v_true).item() * M.size(0)
        total_loss /= dataset_len
        model.train()
        return total_loss
    
    # ==========================================
    # STAGE 1: ADAM TRAINING WITH EARLY STOPPING
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 1: Adam Optimizer")
    print("="*60)
    
    epochs_no_improve = 0
    phase1_epochs_completed = 0
    
    for epoch in range(config['epochs_adam']):
        model.train()
        epoch_start = time.time()
        
        total_loss = 0
        loss_components = {'data': 0, 'pde': 0, 'arb': 0, 'term': 0, 'lower': 0, 'upper': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            M, tau, u, r, q, v_true, mask = [b.to(device) for b in batch]
            M = M.clone().requires_grad_(True)
            tau = tau.clone().requires_grad_(True)
            
            optimizer_adam.zero_grad()
            loss, loss_dict = pinn_loss_fn(
                model, M, tau, u, r, q, v_true, mask,
                lambdas, device,
                num_boundary=config['num_boundary'],
                M_max=config['M_max'],
                tau_max=config['tau_max'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer_adam.step()
            
            total_loss += loss_dict['total']
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Batch Loss: {loss_dict['total']:.6f}")
        
        # Validation
        val_loss = evaluate_model(val_loader, len(val_ds)) # <-- Using generalized helper
        scheduler.step(val_loss)
        current_lr = optimizer_adam.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        
        # Save epoch history
        all_train_losses.append(avg_loss)
        all_val_losses.append(val_loss)
        phase1_epochs_completed += 1
        
        # Average loss components
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        # Check for improvement
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'best_model_adam.pth')
            improved = " *** New Best ***"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"\nAdam Epoch {epoch+1}/{config['epochs_adam']} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        print(f"  Train Loss: {avg_loss:.6f} | Val MSE: {val_loss:.6f}{improved}")
        print(f"  Components - Data: {loss_components['data']:.4f} | PDE: {loss_components['pde']:.4f} | "
              f"Arb: {loss_components['arb']:.4f}")
        print(f"              Term: {loss_components['term']:.4f} | Lower: {loss_components['lower']:.4f} | "
              f"Upper: {loss_components['upper']:.4f}")
        print(f"  Epochs without improvement: {epochs_no_improve}/{config['patience_adam']}")
        print("-" * 60)
        
        # Early stopping check
        if epochs_no_improve >= config['patience_adam']:
            print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
            break
        torch.cuda.empty_cache()
    
    print(f"\n*** Adam Phase Complete. Best Val MSE: {best_val:.6f} ***")
    
    # Load best model before L-BFGS (Fine-Tuning with Adam)
    print("Loading best model from Adam phase for Fine-tuning...")
    model.load_state_dict(torch.load('best_model_adam.pth'))
    
    #changed
    torch.save(model.state_dict(), f'best_model.pth')

    # ==========================================
    # STAGE 2: FINE-TUNING WITH LOW LR ADAM
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning with Adam (Low LR)")
    print("="*60)

    optimizer_finetune = optim.Adam(model.parameters(), lr=1e-4)
    epochs_no_improve = 0

    for epoch in range(config['epochs_lbfgs']):
        model.train()
        epoch_start = time.time()
        
        total_loss = 0
        loss_components = {'data': 0, 'pde': 0, 'arb': 0, 'term': 0, 'lower': 0, 'upper': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            M, tau, u, r, q, v_true, mask = [b.to(device) for b in batch]
            M = M.clone().requires_grad_(True)
            tau = tau.clone().requires_grad_(True)
            
            optimizer_finetune.zero_grad()
            loss, loss_dict = pinn_loss_fn(
                model, M, tau, u, r, q, v_true, mask,
                lambdas, device,
                num_boundary=config['num_boundary'],
                M_max=config['M_max'],
                tau_max=config['tau_max'],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer_finetune.step()
            
            total_loss += loss_dict['total']
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Batch Loss: {loss_dict['total']:.6f}")
        
        val_loss = evaluate_model(val_loader, len(val_ds))
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        
        # Save epoch history
        all_train_losses.append(avg_loss)
        all_val_losses.append(val_loss)
        
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            improved = " *** New Best ***"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"\nFine-tune Epoch {epoch+1}/{config['epochs_lbfgs']} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_loss:.6f} | Val MSE: {val_loss:.6f}{improved}")
        print(f"  Components - Data: {loss_components['data']:.4f} | PDE: {loss_components['pde']:.4f} | "
            f"Arb: {loss_components['arb']:.4f}")
        print(f"              Term: {loss_components['term']:.4f} | Lower: {loss_components['lower']:.4f} | "
            f"Upper: {loss_components['upper']:.4f}")
        print("-" * 60)
        
        torch.cuda.empty_cache()
        
        if epochs_no_improve >= config['patience_lbfgs']:
            print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best Validation MSE: {best_val:.6f}")
    print("="*60)
    
    torch.save(model.state_dict(), 'final_model.pth')
    print("Models saved: 'best_model.pth' and 'final_model.pth'")
    
    # ==========================================
    # EVALUATE TEST SET & SAVE RESULTS
    # ==========================================
    print("\nEvaluating best model on Test Set...")
    model.load_state_dict(torch.load('best_model.pth')) # Load best weights
    test_loss = evaluate_model(test_loader, len(test_ds))
    print(f"Test Set MSE: {test_loss:.6f}")

    # 1. Save to TXT file
    print("\nSaving loss history to 'loss_history.txt'...")
    with open('loss_history.txt', 'w') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\n")
        for i, (t_loss, v_loss) in enumerate(zip(all_train_losses, all_val_losses)):
            f.write(f"{i+1}\t{t_loss:.6f}\t{v_loss:.6f}\n")
        f.write(f"\nFinal Test Loss: {test_loss:.6f}\n")
        
    # 2. Generate and save plot
    print("Generating loss curve plot 'loss_plot.png'...")
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label='Train Loss (Total)')
    plt.plot(all_val_losses, label='Validation Loss (MSE)')
    
    # Mark the transition between Phase 1 and Phase 2
    if phase1_epochs_completed < len(all_train_losses):
        plt.axvline(x=phase1_epochs_completed - 1, color='r', linestyle='--', label='Start Fine-Tuning')
        
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Loss (Test MSE: {test_loss:.6f})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('loss_plot.png')
    plt.close()

    print("Execution Finished.")
    return model

if __name__ == "__main__":
    train_model()