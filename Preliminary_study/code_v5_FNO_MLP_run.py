"""
FNO-DeepONet Black-Scholes Solver
Optimized for 2D Grid branch data (u: 11 days x 34 deltas) and continuous trunk coordinates (M, tau)
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
import matplotlib.pyplot as plt

# ==========================================
# NETWORK ARCHITECTURE (FNO-DeepONet)
# ==========================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute 2D Real Fast Fourier Transform
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        
        # Lower Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        
        # Upper Fourier modes
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to spatial domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO_Branch(nn.Module):
    def __init__(self, modes1=4, modes2=12, width=32, p_dim=128):
        super().__init__()
        self.modes1 = modes1 # Truncated modes for 'days'
        self.modes2 = modes2 # Truncated modes for 'deltas'
        self.width = width
        
        # Lifting layer (1 channel -> width channels)
        self.fc0 = nn.Conv2d(1, self.width, 1) 
        
        # FNO Blocks
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # 1x1 Convolution bypasses for residual connections
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Projection to DeepONet basis dimension
        # The flattened grid size will be width * 11 * 34
        self.fc1 = nn.Linear(self.width * 11 * 34, 256)
        self.fc2 = nn.Linear(256, p_dim)

    def forward(self, x):
        # Reshape flat 374 vector into 2D Grid: (batch, channels, days, deltas)
        x = x.view(-1, 1, 11, 34)
        
        x = self.fc0(x)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)
        
        # Flatten and project
        x = x.view(x.shape[0], -1) 
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


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


class FNO_DeepONet(nn.Module):
    def __init__(self, hidden_dim, num_layers, p_dim, fno_width=32):
        super().__init__()
        # 1. New FNO Branch
        self.branch = FNO_Branch(
            modes1=4, 
            modes2=12, 
            width=fno_width, 
            p_dim=p_dim
        )
        
        # 2. Standard MLP Trunk for continuous variables (M, tau)
        self.trunk = PINN_MLP(
            input_dim=2, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=p_dim, 
            is_positive=False
        )
        
        self.bias = nn.Parameter(torch.zeros(1))
        self.softplus = nn.Softplus()

    def forward(self, u, M, tau):
        branch_out = self.branch(u)
        
        trunk_input = torch.cat([M, tau], dim=1)
        trunk_out = self.trunk(trunk_input)
        
        # Compute the dot product across the basis dimension (p_dim)
        dot_product = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        
        # Add bias and enforce positivity for volatility
        sigma = self.softplus(dot_product + self.bias)
        return sigma


class DeepBS_Solver(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, p_dim=128):
        super().__init__()
        
        # Network 1 is now an FNO-DeepONet
        self.nn1 = FNO_DeepONet(
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            p_dim=p_dim,
            fno_width=32
        )
        
        # Network 2 remains a standard PINN_MLP predicting option price
        self.nn2 = PINN_MLP(
            input_dim=5, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=1, 
            is_positive=True
        )

    def forward(self, M, tau, u, r, q):
        sigma = self.nn1(u, M, tau)
        nn2_input = torch.cat([M, tau, r, q, sigma], dim=1)
        v_hat = self.nn2(nn2_input)
        
        return sigma, v_hat


# ==========================================
# LOSS FUNCTIONS (Unchanged)
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

    M_term = torch.rand(batch_size, 1, device=device) * M_max
    tau_term = torch.zeros(batch_size, 1, device=device)

    M_lower = torch.zeros(batch_size, 1, device=device)
    tau_lower = torch.rand(batch_size, 1, device=device) * tau_max

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
    # Boundary Conditions 
    # ----------------
    boundaries = generate_boundary_points(
        num_boundary, u, r, q, device, M_max=M_max, tau_max=tau_max
    )

    M_t, tau_t, u_t, r_t, q_t = boundaries['terminal']
    M_t.requires_grad_(True)
    tau_t.requires_grad_(True)
    _, v_term = model(M_t, tau_t, u_t, r_t, q_t)
    loss_term = F.mse_loss(v_term, F.relu(M_t - 1.0))

    M_l, tau_l, u_l, r_l, q_l = boundaries['lower']
    M_l.requires_grad_(True)
    tau_l.requires_grad_(True)
    _, v_lower = model(M_l, tau_l, u_l, r_l, q_l)
    loss_lower = F.mse_loss(v_lower, torch.zeros_like(v_lower))

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
# DATASET & COLLATOR (Unchanged)
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
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = {
        'epochs_adam': 100,      
        'epochs_lbfgs': 50, 
        'batch_size': 256,  #changed
        'lr': 1e-3,
        'hidden_dim': 64,
        'num_layers': 4,
        'p_dim': 64,             
        'colloc_ratio': 0.2,
        'points_per_u': 20,      #changed
        'num_boundary': 128, 
        'grad_clip': 1.0,
        'M_max': 3.0,
        'tau_max': 5.32,
        'patience_adam': 3,     #changed
        'patience_lbfgs': 3,    #changed
    }
    
    #changed
    lambdas = {'data': 1.0, 'pde': 0.5, 'arb': 0.5, 'term': 0.5, 'lower': 0.5, 'upper': 0.5}
    
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
            points_per_u=config['points_per_u']
        )
    )
    
    val_loader = DataLoader(
        val_ds,
        config['batch_size'] * 2,
        shuffle=False,
        collate_fn=create_pinn_collate_fn(
            colloc_ratio=0.0,
            M_max=config['M_max'],
            tau_max=config['tau_max']
        )
    )

    test_loader = DataLoader(
        test_ds, config['batch_size'] * 2, shuffle=False,
        collate_fn=create_pinn_collate_fn(colloc_ratio=0.0, M_max=config['M_max'], tau_max=config['tau_max'])
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model (Now using FNO_DeepONet implicitly via DeepBS_Solver changes)
    model = DeepBS_Solver(config['hidden_dim'], config['num_layers'], config['p_dim']).to(device)
    print(f"FNO-DeepONet Model initialized with hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}, p_dim={config['p_dim']}")
    
    # ==========================================
    # OPTIMIZER SETTINGS
    # ==========================================
    base_lr = config['lr']
    
    optimizer_adam = optim.Adam([
        {'params': model.nn1.branch.parameters(), 'lr': base_lr, 'weight_decay': 1e-4},
        {'params': model.nn1.trunk.parameters(), 'lr': base_lr * 0.1, 'weight_decay': 0.0},
        {'params': [model.nn1.bias], 'lr': base_lr},
        {'params': model.nn2.parameters(), 'lr': base_lr, 'weight_decay': 1e-5}
    ])

    #changed
    scheduler = ReduceLROnPlateau(optimizer_adam, 'min', factor=0.5, patience=2)
    
    best_val = float('inf')
    all_train_losses = []
    all_val_losses = []
    
    def evaluate_validation():
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
                _, v_hat = model(M, tau, u, r, q)
                val_loss += F.mse_loss(v_hat, v_true).item() * M.size(0)
        val_loss /= len(val_ds)
        model.train()
        return val_loss
    
    # ==========================================
    # STAGE 1: ADAM TRAINING
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 1: Adam Optimizer (FNO Parameter Groups)")
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
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Batch Loss: {loss_dict['total']:.6f}")
        
        val_loss = evaluate_validation()
        scheduler.step(val_loss)
        current_lr = optimizer_adam.param_groups[0]['lr'] 
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        
        all_train_losses.append(avg_loss)
        all_val_losses.append(val_loss)
        phase1_epochs_completed += 1
        
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'best_model_phase1.pth')
            improved = " *** New Best ***"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"\nAdam Epoch {epoch+1}/{config['epochs_adam']} | Time: {epoch_time:.1f}s | Base LR: {current_lr:.2e}")
        print(f"  Train Loss: {avg_loss:.6f} | Val MSE: {val_loss:.6f}{improved}")
        print(f"  Components - Data: {loss_components['data']:.4f} | PDE: {loss_components['pde']:.4f} | Arb: {loss_components['arb']:.4f}")
        print(f"  Epochs without improvement: {epochs_no_improve}/{config['patience_adam']}")
        print("-" * 60)
        
        if epochs_no_improve >= config['patience_adam']:
            print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
            break
            
        torch.cuda.empty_cache()
    
    print(f"\n*** Adam Phase Complete. Best Val MSE: {best_val:.6f} ***")
    
    print("Loading best model from Adam phase for fine-tuning...")
    model.load_state_dict(torch.load('best_model_phase1.pth'))

    #changed
    torch.save(model.state_dict(), f'best_model.pth')
    
    # ==========================================
    # STAGE 2: FINE-TUNING WITH LOW LR ADAM
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning with Adam (Low LR)")
    print("="*60)

    finetune_lr = 1e-5
    optimizer_finetune = optim.Adam([
        {'params': model.nn1.branch.parameters(), 'lr': finetune_lr, 'weight_decay': 1e-5},
        {'params': model.nn1.trunk.parameters(), 'lr': finetune_lr * 0.1, 'weight_decay': 0.0},
        {'params': [model.nn1.bias], 'lr': finetune_lr},
        {'params': model.nn2.parameters(), 'lr': finetune_lr, 'weight_decay': 1e-6}
    ])
    
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
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Batch Loss: {loss_dict['total']:.6f}")
        
        val_loss = evaluate_validation()
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        
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
        print("-" * 60)
        
        torch.cuda.empty_cache()
        
        if epoch >= 10 and epochs_no_improve >= config['patience_lbfgs']:
            print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best Validation MSE: {best_val:.6f}")
    print("="*60)
    
    torch.save(model.state_dict(), 'final_model.pth')
    
    # ==========================================
    # FINAL TEST EVALUATION
    # ==========================================
    print("\n" + "="*60)
    print("RUNNING FINAL TEST SET EVALUATION")
    print("="*60)
    
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
            _, v_hat = model(M, tau, u, r, q)
            test_loss += F.mse_loss(v_hat, v_true).item() * M.size(0)
            
    test_loss /= len(test_ds)
    print(f"Final Test MSE: {test_loss:.6f}")
    
    # ==========================================
    # EXPORTING LOGS & PLOTTING
    # ==========================================
    print("\nSaving loss history to 'loss_history.txt'...")
    with open('loss_history.txt', 'w') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\n")
        for i, (t_loss, v_loss) in enumerate(zip(all_train_losses, all_val_losses)):
            f.write(f"{i+1}\t{t_loss:.6f}\t{v_loss:.6f}\n")
        f.write(f"\nFinal Test Loss: {test_loss:.6f}\n")

    print("Generating loss curve plot 'loss_plot.png'...")
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label='Train Loss (Total PINN Loss)')
    plt.plot(all_val_losses, label='Validation Loss (MSE)')
    
    if phase1_epochs_completed < len(all_train_losses):
        plt.axvline(x=phase1_epochs_completed - 1, color='r', linestyle='--', label='Start Phase 2 (Fine-tuning)')
        
    plt.yscale('log')
    plt.xlabel('Epoch')
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