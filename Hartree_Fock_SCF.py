import numpy as np
import time

# ============================================================
# Helper functions
# ============================================================

def one_elec_int(filename):
    """Read symmetric matrix (1-based indices)."""
    lines = open(filename).readlines()
    n = max(int(l.split()[0]) for l in lines)
    M = np.zeros((n, n))
    for line in lines:
        i, j, val = line.split()
        i, j = int(i) - 1, int(j) - 1
        M[i, j] = M[j, i] = float(val)
    return M

def two_elec_int(filename):
    """ Reag 4D two-electron tensor (1-based indices). """
    lines = open(filename).readlines()
    n = 7
    eri = np.zeros((n, n, n, n))
    for line in lines:
        i, j, k, l, val = line.split()
        i, j, k, l = int(i) - 1, int(j) - 1, int(k) - 1, int(l) - 1
        val = float(val)
        eri[i, j, k, l] = eri[j, i, k, l] = eri[i, j, l, k] = eri[j, i, l, k] = val
        eri[k, l, i, j] = eri[l, k, i, j] = eri[k, l, j, i] = eri[l, k, j, i] = val
    return eri


def den_mat(sqrt_S, F, nocc=5):  # nocc is number of occupied MO
    """Build density matrix from Fock and overlap matrices."""
    Fp = sqrt_S.T @ F @ sqrt_S
    _, Cp = np.linalg.eigh(Fp)
    C = sqrt_S @ Cp
    D = C[:, :nocc] @ C[:, :nocc].T
    return D


def error(D_value, iteration):
    """Compute RMS difference between two densities."""
    diff = D_value[iteration] - D_value[iteration - 1]
    return np.sqrt(np.sum(diff**2))


# ============================================================
# Efficient Fock Builder
# ============================================================

def build_fock(H, D, two_electron_tensor):
    """
    Construct Fock matrix:
        F_ij = H_ij + Σ_kl D_kl (2⟨ij|kl⟩ - ⟨ik|jl⟩)
    two_electron_tensor should be 4D array [i,j,k,l].
    """
    # Compute Coulomb and Exchange terms efficiently using tensor contractions
    J = np.einsum("kl,ijkl->ij", D, two_electron_tensor, optimize=True)
    K = np.einsum("kl,ikjl->ij", D, two_electron_tensor, optimize=True)
    F = H + 2 * J - K
    return F


# ============================================================
# Main SCF Procedure
# ============================================================

def hartree_fock_scf():
    tic = time.time()

    # ---------- Read integrals ----------
    e_nuc = float(open("nuclear_repulsion_energy.dat").read().strip())
    S = one_elec_int("overlap_matrix.dat")
    T = one_elec_int("kinetic_energy_matrix.dat")
    V = one_elec_int("nuclear_attraction_energy_matrix.dat")
    H = T + V
    eri = two_elec_int("two_electron_integral.dat")

    # ---------- Orthogonalization ----------
    eigval, eigvec = np.linalg.eigh(S)
    sqrt_S = eigvec @ np.diag(eigval**-0.5) @ eigvec.T

    # ---------- Initial Guess ----------
    D0 = den_mat(sqrt_S, H)
    D_value = [D0]
    E0_elc = np.sum(D0 * (H + H))
    E_total = E0_elc + e_nuc
    E = [E_total]
    print(f"Initial total energy: {E_total: .12f}")

    iteration, delta_E, rms_D = 1, 1.0, 1.0

    # ---------- SCF Loop ----------
    while delta_E > 1e-12 or rms_D > 1e-10:
        F = build_fock(H, D_value[-1], eri)
        D_new = den_mat(sqrt_S, F)
        D_value.append(D_new)

        E_elc = np.sum(D_new * (H + F))
        E_total = E_elc + e_nuc
        delta_E = abs(E_total - E[-1])
        rms_D = error(D_value, iteration)

        print(f"{iteration:2d}  {E_elc:20.12f}  {E_total:20.12f}  ΔE={delta_E:.3e}  RMS={rms_D:.3e}")
        E.append(E_total)
        iteration += 1

    print(f"\nE_SCF = {E_total: .12f}")
    print("Expected E_SCF ≈ -74.942079928192")
    print("Elapsed time = {:.4f} s".format(time.time() - tic))


# ============================================================
# Run the program
# ============================================================
if __name__ == "__main__":
    hartree_fock_scf()

