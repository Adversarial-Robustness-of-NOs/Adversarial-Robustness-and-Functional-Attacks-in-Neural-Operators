import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# Shared Constants
# ==========================================
L = 1.0             # Domain size
CFL = 0.25          # Stability factor (Still used for time-stepping checks)
T_MAX = 2.0         # Simulation time (if using time-stepping)

# ==========================================
# Core Physics Functions
# ==========================================

def bc_neumann(u):
    """
    Sets boundary ghost cells to 0.0 (Dirichlet-like behavior at walls).
    Pads input (N, N) -> (N+4, N+4).
    """
    return F.pad(u, (2, 2, 2, 2), mode='constant', value=0.0)

def darcy_operator(u, nu_padded, dx, dy):
    """
    Computes the operator A(u) = - div(nu * grad(u)).
    This is the LHS of the steady state equation: -div(nu*grad(u)) = force.
    """
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy
    
    # Apply BC (ghost cells = 0)
    u_padded = bc_neumann(u)
    
    # --- Flux Calculation ---
    # Flux X: -nu * du/dx
    # nu_padded is pre-padded, so we slice into it directly
    nu_x = 0.5 * (nu_padded[2:-1, 2:-2] + nu_padded[1:-2, 2:-2])
    du_x = (u_padded[2:-1, 2:-2] - u_padded[1:-2, 2:-2]) * dx_inv
    fx = -nu_x * du_x
    
    # Flux Y: -nu * du/dy
    nu_y = 0.5 * (nu_padded[2:-2, 2:-1] + nu_padded[2:-2, 1:-2])
    du_y = (u_padded[2:-2, 2:-1] - u_padded[2:-2, 1:-2]) * dy_inv
    fy = -nu_y * du_y
    
    # --- Divergence of Flux ---
    # The PDE is -div(flux) = force
    # Since we defined flux = -nu*grad, div(flux) = -div(nu*grad).
    # We want to return -div(nu*grad), which is exactly div(flux).
    div_fx = (fx[1:, :] - fx[:-1, :]) * dx_inv
    div_fy = (fy[:, 1:] - fy[:, :-1]) * dy_inv
    
    return div_fx + div_fy

def update_step(u, dt, nu_padded, dx, dy, force):
    """
    Original Explicit Time Step (Slow). kept for backward compatibility.
    u_new = u - dt * (A(u) - force)
    """
    Au = darcy_operator(u, nu_padded, dx, dy)
    return u - dt * (Au - force)

# ==========================================
# High-Level Interfaces
# ==========================================

def generate_permeability(x_resolution, y_resolution, device='cpu', seed=None):
    """
    Generates a single sample of the 'Islands' permeability map.
    Replicates PDEBench logic: Sum of Gaussians -> Threshold.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    # Grid
    x = torch.linspace(0, L, x_resolution, device=device)
    y = torch.linspace(0, L, y_resolution, device=device)
    xc, yc = torch.meshgrid(x, y, indexing='ij')
    
    # 5 random centers and widths
    xms = torch.rand(5, device=device)
    yms = torch.rand(5, device=device)
    stds = 0.5 * torch.rand(5, device=device)
    
    nu = torch.zeros((x_resolution, y_resolution), device=device)
    
    for i in range(5):
        dist_sq = (xc - xms[i])**2 + (yc - yms[i])**2
        nu += torch.exp(-dist_sq / stds[i])
        
    # Thresholding
    mean_val = torch.mean(nu)
    nu = torch.where(nu > mean_val, 
                     torch.tensor(1.0, device=device), 
                     torch.tensor(0.1, device=device))
    return nu

def solve_steady_state_cg(nu, force=1.0, max_iter=2000, tol=1e-6):
    """
    Solves -div(nu * grad(u)) = force using Conjugate Gradient.
    This is ~500x faster than explicit time stepping.
    """
    x_res = nu.shape[0]
    y_res = nu.shape[1]
    dx = L / (x_res - 1)
    dy = L / (y_res - 1)
    nu_padded = bc_neumann(nu)
    
    # RHS vector (Force term)
    b = torch.ones_like(nu) * force
    
    # Initial Guess (u=0)
    x = torch.zeros_like(nu)
    
    # Calculate initial residual: r = b - Ax
    # Since x=0, Ax=0, so r = b
    r = b.clone()
    p = r.clone()
    rsold = torch.sum(r * r)
    
    for i in range(max_iter):
        # Calculate A * p
        Ap = darcy_operator(p, nu_padded, dx, dy)
        
        # Alpha: Step size
        alpha = rsold / (torch.sum(p * Ap) + 1e-10)
        
        # Update Solution x and Residual r
        x = x + alpha * p
        r = r - alpha * Ap
        
        # Check convergence
        rsnew = torch.sum(r * r)
        if torch.sqrt(rsnew) < tol:
            # print(f"CG converged in {i} iterations.")
            break
            
        # Beta: Update direction
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
        
    return x

# Wrapper to maintain API compatibility
def solve_steady_state(nu, force=1.0):
    return solve_steady_state_cg(nu, force)

# ==========================================
# Main Test Block
# ==========================================
if __name__ == "__main__":
    print("Testing Darcy Solver (Fast CG)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Generation
    nu = generate_permeability(128, 128, device=device)
    
    # Test Solver
    import time
    start = time.time()
    u = solve_steady_state(nu, force=1.0)
    end = time.time()
    
    print(f"Solved in {end - start:.4f} seconds!")
    print(f"u min: {u.min():.4f}, u max: {u.max():.4f}")
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(nu.cpu(), origin='lower'); plt.title("Nu")
    plt.subplot(1,2,2); plt.imshow(u.cpu(), origin='lower'); plt.title("U")
    plt.show()