import numpy as np
from ase.io import read
import matplotlib.pyplot as plt 
from ase import Atoms
from ase.build import graphene
from ase.visualize import view
from ase.build import graphene_nanoribbon
from ase.neighborlist import neighbor_list
from ase.build import make_supercell

# Ase
def agnr(N, L):
    """
    Create an armchair graphene nanoribbon.
    
    Parameters
    ----------
    N : int
        Width parameter of the nanoribbon
    L : int
        Length parameter of the nanoribbon
    """
    rib = graphene_nanoribbon(
        n=N,
        m=L,
        type='armchair',
        saturated=False,
        C_C=1.42,
        vacuum=12,
    )
    return rib

def zgnr(N, L):
    """
    Create a zigzag graphene nanoribbon.
    
    Parameters
    ----------
    N : int
        Width parameter of the nanoribbon
    L : int
        Length parameter of the nanoribbon
    """
    rib = graphene_nanoribbon(
        n=N,
        m=L,
        type='zigzag',
        saturated=False,
        C_C=1.42,
        vacuum=12,
    )
    return rib

def repeat_str(atoms, n, axis=0):
    """
    Repeat atomic structure along a specified axis.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Input atomic structure
    n : int
        Number of repetitions
    axis : int, optional
        Axis along which to repeat (0=x, 1=y, 2=z), default is 0
    
    Returns
    -------
    ase.Atoms
        Repeated atomic structure with updated cell
    """
    cell = atoms.cell
    shift_vec = cell[axis]  # lattice vector along which to repeat

    all_positions = []
    all_numbers = []

    for i in range(n):
        shift = i * shift_vec
        all_positions.append(atoms.positions + shift)
        all_numbers.extend(atoms.numbers)

    all_positions = np.vstack(all_positions)

    new_cell = cell.copy()
    new_cell[axis] *= n

    return Atoms(
        numbers=all_numbers,
        positions=all_positions,
        cell=new_cell,
        pbc=atoms.pbc
    )

def add_AA_on_top(atoms, distance=3.35):
    """
    Add a second layer on top in AA stacking configuration.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Bottom layer atomic structure
    distance : float, optional
        Vertical separation between layers in Angstrom, default is 3.35
    
    Returns
    -------

    ase.Atoms    return stacked

        AA-stacked bilayer structure

    """
    top = atoms.copy()

    top.positions[:, 2] += distance

    stacked = atoms + top

    cell = stacked.cell.copy()
    if cell[2, 2] < distance + 10:
        cell[2, 2] = distance + 10
    stacked.set_cell(cell)

    return stacked

def add_AB_on_top(atoms, distance=3.35):
    """
    Add a second GNR on top in AB (Bernal) stacking configuration.
    Assumes the ribbon lies in the xy plane.
    <
    Parameters
    ----------
    atoms : ase.Atoms
        Bottom layer atomic structure
    distance : float, optional
        Vertical separation between layers in Angstrom, default is 3.35
    
    Returns
    -------
    ase.Atoms
        AB-stacked bilayer structure with atoms wrapped in unit cell
    """

    # AB shift in xy
    dx = 1.43
    dy = 0

    # Copy bottom layer
    top = atoms.copy()

    # Apply AB shift + vertical separation
    top.positions[:, 0] += dx
    top.positions[:, 1] += dy
    top.positions[:, 2] += distance

    # Wrap top-layer atoms back into the unit cell (x and y only)
    cell = atoms.cell
    Lx = cell[0, 0]
    Ly = cell[1, 1]

    pos = top.positions
    pos[:, 0] = np.mod(pos[:, 0], Lx)
    pos[:, 1] = np.mod(pos[:, 1], Ly)
    top.positions = pos

    # Merge layers
    stacked = atoms + top

    # Ensure enough vacuum in z
    new_cell = cell.copy()
    if new_cell[2, 2] < distance + 10:
        new_cell[2, 2] = distance + 10
    stacked.set_cell(new_cell)

    return stacked


# Graphene
def make_graphene_monolayer(nx=4, ny=4, a=2.46):
    """
    Create a graphene monolayer supercell in the xy plane.
    
    Parameters
    ----------
    nx : int, optional
        Number of unit cells along x direction, default is 4
    ny : int, optional
        Number of unit cells along y direction, default is 4
    a : float, optional
        Lattice constant in Angstrom, default is 2.46
    
    Returns
    -------
    ase.Atoms
        Graphene monolayer structure with periodic boundary conditions in xy
    """
    # Primitive lattice vectors in xy
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a/2, np.sqrt(3)*a/2, 0.0])

    # Supercell lattice
    cell = np.array([
        nx * a1,
        ny * a2,
        [0.0, 0.0, 20.0]  # vacuum in z
    ])

    # Basis positions (two atoms per unit cell)
    basis = np.array([
        [0.0,           0.0,           0.0],
        [a/2, np.sqrt(3)*a/6,          0.0],
    ])

    positions = []
    numbers = []

    for i in range(nx):
        for j in range(ny):
            R = i * a1 + j * a2
            for b in basis:
                positions.append(R + b)
                numbers.append(6)  # carbon

    return Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True, True, False]
    )

def make_AA_bilayer_graphene(nx=4, ny=4, a=2.46, dz=3.35):

    bottom = make_graphene_monolayer(nx, ny, a=a)

    top = bottom.copy()
    top.positions[:, 2] += dz  # shift only in z

    bilayer = bottom + top

    cell = bilayer.cell.copy()
    if cell[2, 2] < dz + 10:
        cell[2, 2] = dz + 10
    bilayer.set_cell(cell)

    return bilayer



def make_AB_bilayer_graphene(nx=4, ny=4, a=2.46, dz=3.35):
    bottom = make_graphene_monolayer(nx, ny, a=a)

    # lattice vectors in xy
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a/2, np.sqrt(3)*a/2, 0.0])

    # Bernal (AB) in-plane shift
    shift_xy = (a1 + a2) / 3.0

    top = bottom.copy()
    top.positions[:, :2] += shift_xy[:2]
    top.positions[:, 2]  += dz

    bilayer = bottom + top

    cell = bilayer.cell.copy()
    if cell[2, 2] < dz + 10:
        cell[2, 2] = dz + 10
    bilayer.set_cell(cell)

    return bilayer
 

# Hamiltons

def hamiltonian(xyz,a = 1.42,Vpppi = -2.7):
    #### Tight binding hamiltonian for a set of atomic coordinates in units of Vpppi
    dist = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    return np.where((dist < (a + 0.15)) & (dist > 0.1), Vpppi, 0)   

def slater_koster(
    atoms,
    cutoff=6,
    d0=1.42,
    d_gg=3.35,
    r0=0.45,
    Vpp_pi0=-2.7,
    Vpp_sigma0=0.48,
    onsite=0.0,
    carbon_only=True
):
    """
    Compute Slater-Koster tight-binding Hamiltonian for graphene structures.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure
    cutoff : float, optional
        Cutoff distance for interlayer hopping in Angstrom, default is 6
    d0 : float, optional
        In-plane C-C bond length in Angstrom, default is 1.42
    d_gg : float, optional
        Interlayer distance in Angstrom, default is 3.35
    r0 : float, optional
        Decay length in Angstrom, default is 0.45
    Vpp_pi0 : float, optional
        Pi-orbital hopping parameter in eV, default is -2.7
    Vpp_sigma0 : float, optional
        Sigma-orbital hopping parameter in eV, default is 0.48
    onsite : float, optional
        On-site energy in eV, default is 0.0
    carbon_only : bool, optional
        Whether to consider only carbon atoms, default is True
    
    Returns
    -------
    H : np.ndarray
        Total Hamiltonian matrix (H_TB + H_SK)
    H_TB : np.ndarray
        In-plane tight-binding Hamiltonian
    H_SK : np.ndarray
        Slater-Koster interlayer Hamiltonian
    H_pi : np.ndarray
        Pi-orbital contribution
    H_sigma : np.ndarray
        Sigma-orbital contribution
    """
    pos = atoms.positions
    N = len(atoms)

    H_TB = np.zeros((N, N))
    H_pi = np.zeros((N, N),)
    H_sigma = np.zeros((N, N))

    # onsite term (pi channel)
    for i in range(N):
        H_pi[i, i] = onsite

    # In plane hopping
    for i in range(N):
        ri = pos[i]
        z_i=pos[i][2]
        for j in range(i + 1, N):
            rj = pos[j]
            z_j=pos[j][2]
            dvec = rj - ri
            R = np.linalg.norm(dvec)
            # check if the two atoms lie on the same z plane
            if np.abs(z_i-z_j) < 0.1: 
                 # Make TB hamiltonian
                 dist = np.linalg.norm(ri -rj, axis=0)
                 if  R < 1.5 and R > 0.1:
                    H_TB[i, j] = Vpp_pi0
                    H_TB[j, i] = Vpp_pi0
            else:
                # Make SK
                if R < cutoff and R > 1e-8:
    
                    # angle factors
                    cos_theta = dvec[2] / R
                    cos2 = cos_theta**2
                    sin2 = 1.0 - cos2
    
                    # distance dependent hoppings
                    Vpi = Vpp_pi0 * np.exp(-(R - d0) / r0)
                    Vsigma = Vpp_sigma0 * np.exp(-(R - d_gg) / r0)
    
                    # separated contributions
                    t_pi = Vpi * sin2
                    t_sigma = Vsigma * cos2
    
                    # fill matrices
                    H_pi[i, j] = t_pi
                    H_pi[j, i] = t_pi
    
                    H_sigma[i, j] = t_sigma
                    H_sigma[j, i] = t_sigma


    H_SK = H_pi + H_sigma
    H = H_TB + H_SK
    return H,H_TB,H_SK,H_pi,H_sigma

def get_V_SK_2D(
    atoms,
    scell_size_x=3,
    scell_size_y=3,
    cutoff=9.0,
    d0=1.42,
    d_gg=3.35,
    r0=0.45,
    Vpp_pi0=-2.7,
    Vpp_sigma0=0.48,
    onsite=0.0,
    axis_0=0,axis_1=1
    
):
    """
    Compute 2D periodic hopping matrices for Bloch Hamiltonian construction.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Unit cell atomic structure
    scell_size_x : int, optional
        Supercell size along first periodic axis, default is 3
    scell_size_y : int, optional
        Supercell size along second periodic axis, default is 3
    cutoff : float, optional
        Cutoff distance for hopping in Angstrom, default is 9.0
    d0, d_gg, r0, Vpp_pi0, Vpp_sigma0, onsite : float, optional
        Slater-Koster parameters (see slater_koster function)
    axis_0 : int, optional
        First periodic axis (0=x, 1=y, 2=z), default is 0
    axis_1 : int, optional
        Second periodic axis (0=x, 1=y, 2=z), default is 1
    
    Returns
    -------
    H0 : np.ndarray
        On-site block Hamiltonian
    Vx_plus : np.ndarray
        Hopping matrix in +x direction
    Vy_plus : np.ndarray
        Hopping matrix in +y direction
    Vx_minus : np.ndarray
        Hopping matrix in -x direction
    Vy_minus : np.ndarray
        Hopping matrix in -y direction
    """
    # Build 2D supercell
    sc_atoms = repeat_str(atoms, scell_size_x, axis=axis_0)
    sc_atoms = repeat_str(sc_atoms, scell_size_y, axis=axis_1)

    # Compute SK Hamiltonian
    H_sc, _, _, _, _ = slater_koster(
        sc_atoms,
        cutoff=cutoff,
        d0=d0,
        d_gg=d_gg,
        r0=r0,
        Vpp_pi0=Vpp_pi0,
        Vpp_sigma0=Vpp_sigma0,
        onsite=onsite
    )

    N = len(atoms)

    # Middle cell index
    mid_x = scell_size_x // 2
    mid_y = scell_size_y // 2

    # Convert (ix, iy) → flat index
    def idx(ix, iy):
        return (iy * scell_size_x + ix) * N

    # Central block
    i0 = idx(mid_x, mid_y)
    i1 = i0 + N
    H0 = H_sc[i0:i1, i0:i1]

    # Couplings in +x, -x
    Vx_plus  = H_sc[i0:i1, idx(mid_x+1, mid_y):idx(mid_x+1, mid_y)+N]
    Vx_minus = H_sc[i0:i1, idx(mid_x-1, mid_y):idx(mid_x-1, mid_y)+N]

    # Couplings in +y, -y
    Vy_plus  = H_sc[i0:i1, idx(mid_x, mid_y+1):idx(mid_x, mid_y+1)+N]
    Vy_minus = H_sc[i0:i1, idx(mid_x, mid_y-1):idx(mid_x, mid_y-1)+N]

    return H0, Vx_plus,Vy_plus,Vx_minus,Vy_minus

def H_k_1D(atoms, H0, V_right, k, axis=0):
    """
    Construct 1D Bloch Hamiltonian at a given k-point.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure
    H0 : np.ndarray
        On-site Hamiltonian block
    V_right : np.ndarray

        Hopping matrix to the right unit cell    return Hk

    k : float

        k-point value in inverse Angstrom    Hk = H0 + V_right * phase + V_right.conj().T * np.conj(phase)

    axis : int, optional    # Bloch Hamiltonian

        Periodic axis direction (0=x, 1=y, 2=z), default is 0

        phase = np.exp(1j * k * a)
    """
    # lattice period along chosen axis
    a = atoms.cell[axis, axis]   # works for orthorhombic cells

    # Bloch phase
    phase = np.exp(1j * k * a)

    # Bloch Hamiltonian
    Hk = H0 + V_right * phase + V_right.conj().T * np.conj(phase)

    return Hk

def H_k_2D(atoms,H0,Vx,Vy,kx,ky,axis_x=0,axis_y=1):  
    """
    Construct 2D Bloch Hamiltonian at a given (kx, ky) point.
    
    Parameters

    ----------    return Hk 

    atoms : ase.Atoms    Hk = H0 + Vx * np.exp(1j*kx*ax) + Vx.conj().T * np.exp(-1j*kx*ax)+ Vy * np.exp(1j*ky*ay) + Vy.conj().T * np.exp(-1j*ky*ay)

        Atomic structure    ay=atoms.cell[axis_y,axis_y]

    H0 : np.ndarray    ax=atoms.cell[axis_x,axis_x]

        On-site Hamiltonian block    
    """  
    ax=atoms.cell[axis_x,axis_x]
    ay=atoms.cell[axis_y,axis_y]
    Hk = H0 + Vx * np.exp(1j*kx*ax) + Vx.conj().T * np.exp(-1j*kx*ax)+ Vy * np.exp(1j*ky*ay) + Vy.conj().T * np.exp(-1j*ky*ay)
    return Hk 

# Alan ex 28

def SplitHam(H, nL, nR):
    """
    Partition Hamiltonian H with block structure:
    L1 | L2 | C | R1 | R2
    Device D = L2 | C | R1
    """

    no = H.shape[0]
    nC = no - 2*nL - 2*nR
    nD = nL + nC + nR

    if nC < 1:
        print("Setup error: central region size =", nC)
        print("Use [L | L | C | R | R] setup")
        return

    # Boundaries
    b0 = 0
    b1 = nL
    b2 = 2*nL
    b3 = 2*nL + nC
    b4 = 2*nL + nC + nR
    b5 = no

    # Left lead onsite (L1)
    HL = H[b0:b1, b0:b1]

    # L2 → L1 hopping
    VL = H[b1:b2, b0:b1]

    # Coupling L2 ↔ C
    VCL = H[b2:b3, b1:b2]   # C → L2
    VLC = VCL.T.conj()      # L2 → C

    # Central region
    HC = H[b2:b3, b2:b3]

    # Coupling C ↔ R1
    VCR = H[b2:b3, b3:b4]   # C → R1
    VRC = VCR.T.conj()      # R1 → C

    # R1 → R2 hopping
    VR = H[b3:b4, b4:b5]

    # Right lead onsite (R2)
    HR = H[b4:b5, b4:b5]

    # Device block
    HD = H[b1:b4, b1:b4]

    # Lead–device couplings
    VLD = H[b1:b4, b0:b1]   # L1 → device
    VRD = H[b1:b4, b4:b5]   # device → R2

    # Return in left → right order
    return HL,VL, HR, VR, HC, VLC, VRC, HD, VLD, VRD   

def get_surface_greens_function(h_unit, v_unit,z, max_iter=100,tol=1e-10):
    """Sancho-Rubio iterative algorithm for lead surface Green's function."""
    h = np.array(h_unit, dtype=complex)
    v = np.array(v_unit, dtype=complex)
    v_dag = v.T.conj()
    dim = h.shape[0]
    I = np.eye(dim)
    
    eps_s, eps = h.copy(), h.copy()
    alpha, beta = v.copy(), v_dag.copy()
    
    for _ in range(max_iter):
        zI_eps = z * I - eps
        # Using solve for better numerical stability than direct inv
        g_alpha = np.linalg.solve(zI_eps, alpha)
        g_beta = np.linalg.solve(zI_eps, beta)
        
        alpha_next = alpha @ g_alpha
        beta_next = beta @ g_beta
        eps_next = eps + alpha @ g_beta + beta @ g_alpha
        eps_s_next = eps_s + alpha @ g_beta
        
        if np.linalg.norm(alpha_next, ord=np.inf) < tol:
            eps_s = eps_s_next
            break
        alpha, beta, eps, eps_s = alpha_next, beta_next, eps_next, eps_s_next

    g_s=np.linalg.inv(z * I - eps_s)
    g_b=np.linalg.inv(z * I - eps)
    sigma_s=eps_s-h.copy()
    sigma_b=eps-h.copy()

    return g_s,g_b,sigma_s,sigma_b

def compute_transmission_caroli(energy, eta,HL,VL, HR, VR, HC, VLC, VRC, HD, VLD, VRD,method=1 ):
    """
    Computes transmission T(E) based on the formulas:
    Sigma_L = V_LD.dag @ g_L @ V_LD
    Sigma_R = V_RD @ g_R @ V_RD.dag
    T(E) = Trace[Gamma_R @ G_D @ Gamma_L @ G_D_dag]
    """
    z = energy + 1j * eta

    # 1. Surface Green's Functions for leads
    gl_s,gl_b,sigmal_s,sigmal_b = get_surface_greens_function(HL, VL,z)
    gr_s,gr_b,sigmar_s,sigmar_b = get_surface_greens_function(HR, VR,z)

    # 2. Self-Energies following Sigma_L = V_LD.dag @ g_L @ V_LD
    if method==1:
        dim = HC.shape[0]
        I= np.eye(dim)

        sigma_L = VLC.T.conj() @ gl_s @ VLC  
        sigma_R = VRC.T.conj() @ gr_s @ VRC
    
        gamma_L = 1j * (sigma_L - sigma_L.T.conj())
        gamma_R = 1j * (sigma_R - sigma_R.T.conj())
        
        g_C = np.linalg.inv( z * I - HC - sigma_L - sigma_R)
        t_matrix = gamma_R @ g_C @ gamma_L @ g_C.T.conj()
            
    # Inverting G_D with D=LCR 

    elif method == 2:
        dim = HD.shape[0]
        I = np.eye(dim, dtype=complex)

        # Infer sizes from blocks
        nL = HL.shape[0]
        nR = HR.shape[0]
        nD = dim
        nC = nD - nL - nR

        h_eff = HD.astype(complex).copy()
        
        sigma_L = sigmal_s
        sigma_R = sigmar_s


        h_eff[0:nL, 0:nL] += sigma_L

        h_eff[nL + nC:nD, nL + nC:nD] += sigma_R

        g = np.linalg.inv(z * I - h_eff)

        gamma_L = np.zeros((dim, dim), dtype=complex)
        gamma_R = np.zeros((dim, dim), dtype=complex)

        gamma_L[0:nL, 0:nL] = 1j * (sigma_L - sigma_L.T.conj())
        gamma_R[nL + nC:nD, nL + nC:nD] = 1j * (sigma_R - sigma_R.T.conj())

        g_C = g
        t_matrix = gamma_R @ g_C @ gamma_L @ g.T.conj()

        
    # 5. Transmission T(E) = Trace[Gamma_R @ G_D @ Gamma_L @ G_D_dag]
  
    transmission= np.trace(t_matrix).real
    return transmission, gl_s, gl_b, gr_s, gr_b, g_C,sigma_L,sigma_R

def compute_transmission_caroli_v2(energy, eta,HL,VL, HR, VR, HC, VLC, VRC, HD, VLD, VRD,method=1 ):
    """
    Computes transmission T(E) based on the formulas:
    Sigma_L = V_LD.dag @ g_L @ V_LD
    Sigma_R = V_RD @ g_R @ V_RD.dag
    T(E) = Trace[Gamma_R @ G_D @ Gamma_L @ G_D_dag]
    """
    z = energy + 1j * eta

    # 1. Surface Green's Functions for leads
    gl_s,gl_b,sigmal_s,sigmal_b = get_surface_greens_function(HL, VL,z)
    gr_s,gr_b,sigmar_s,sigmar_b = get_surface_greens_function(HR, VR,z)

    if method == 2:

        # Infer sizes from blocks
        nL = HL.shape[0]
        nR = HR.shape[0]
        nC = HC.shape[0]
        dim = nL+nC+nR # of bottom device part
        

        nD = HD.shape[0]
        I = np.eye(nD, dtype=complex)

        h_eff = HD.astype(complex).copy()
        
        sigma_L = sigmal_s
        sigma_R = sigmar_s


        h_eff[0:nL, 0:nL] += sigma_L

        h_eff[nL + nC:dim, nL + nC:dim] += sigma_R #

        g = np.linalg.inv(z * I - h_eff)

        gamma_L = np.zeros((nD, nD), dtype=complex)
        gamma_R = np.zeros((nD, nD), dtype=complex)

        gamma_L[0:nL, 0:nL] = 1j * (sigma_L - sigma_L.T.conj())
        gamma_R[nL + nC:dim, nL + nC:dim] = 1j * (sigma_R - sigma_R.T.conj())

        g_C = g
        t_matrix = gamma_R @ g_C @ gamma_L @ g.T.conj()

        
    # 5. Transmission T(E) = Trace[Gamma_R @ G_D @ Gamma_L @ G_D_dag]
  
    transmission= np.trace(t_matrix).real
    return transmission, gl_s, gl_b, gr_s, gr_b, g_C,sigma_L,sigma_R

def SplitHamMoy(H, nL, nR, nF):
    """
    Partition Hamiltonian H with block structure:
    L1 | L2 | C | R1 | R2
    Device D = L2 | C | R1
    """

    no = H.shape[0]
    nC = no - 2*nL - 2*nR - nF

    if nC < 1:
        print("Setup error: central region size =", nC)
        print("Use [L | L | C | R | R] setup")
        return

    # Boundaries
    b0 = 0
    b1 = nL
    b2 = 2*nL
    b3 = 2*nL + nC
    b4 = 2*nL + nC + nR
    b5 = 2*nL + nC + 2*nR
    b6 = no

    # Left lead onsite (L1)
    HL = H[b0:b1, b0:b1]

    # L2 → L1 hopping
    VL = H[b1:b2, b0:b1]

    # Coupling L2 ↔ C
    VCL = H[b2:b3, b1:b2]   # C → L2
    VLC = VCL.T.conj()      # L2 → C

    # Central region
    HC = H[b2:b3, b2:b3]

    # Coupling C ↔ R1
    VCR = H[b2:b3, b3:b4]   # C → R1
    VRC = VCR.T.conj()      # R1 → C

    # R1 → R2 hopping
    VR = H[b3:b4, b4:b5]

    # Right lead onsite (R2)
    HR = H[b4:b5, b4:b5]

    # Device block
    rows = np.r_[b1:b4, b5:b6]
    cols = np.r_[b1:b4, b5:b6]

    HD = H[np.ix_(rows, cols)]

    HF = H[b5:b6, b5:b6]

    # Lead–device couplings
    VLD = H[b1:b4, b0:b1]   # L1 → device
    VRD = H[b1:b4, b4:b5]   # device → R2

    # Return in left → right order
    return HL,VL, HR, VR, HC, VLC, VRC, HD, VLD, VRD, HF