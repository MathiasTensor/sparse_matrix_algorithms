# Interior Eigenvalue Solvers (FEAST, Zolotarev, Chebyshev, POLFED)

Four algorithms for computing interior eigenpairs of large sparse or matrix-free operators.

All methods solve the eigenvalue problem

\[
A x = \lambda x
\]

and target eigenvalues inside a prescribed interval

\[
\lambda \in [\mathrm{emin}, \mathrm{emax}].
\]

They are designed for:
- large sparse matrices
- matrix-free operators
- problems with many interior eigenvalues

Each file is self-contained and can be used independently.

---

## Algorithms Overview

### 1. FEAST

FEAST computes interior eigenpairs by approximating the spectral projector

\[
P = \frac{1}{2\pi i}\oint_\Gamma (zI-A)^{-1}\,dz
\]

via a quadrature rule on a complex contour:

\[
P \approx \sum_{k} w_k (z_k I - A)^{-1}.
\]

The filtered subspace is then orthogonalized and a Rayleigh–Ritz projection is performed.

#### High-level API

- `general_feast_sparse_interval(...)`
- `sym_feast_sparse_interval_half(...)`
- `general_feast_sparse_interval_distributed(...)`
- `sym_feast_sparse_interval_half_distributed(...)`

#### When to use
- need many interior eigenvalues
- shifted linear solves are available

For symmetric problems, the half-contour variant reduces cost by ~2×.

> In most cases, prefer Zolotarev FEAST below — it is typically faster and sharper.

---

### 2. Zolotarev FEAST

This replaces the contour quadrature with a Zolotarev rational filter, giving

\[
P \approx \alpha_0 I + \sum_k \frac{w_k}{z_k I - A}.
\]

The poles and residues are chosen optimally to separate the interval from the rest of the spectrum.

#### High-level API

- `general_feast_sparse_interval_zolotarev(...)`
- `general_feast_sparse_interval_zolotarev_distributed(...)`
- `sym_feast_sparse_interval_half_zolotarev(...)`
- `sym_feast_sparse_interval_half_zolotarev_distributed(...)`
- `sym_feast_sparse_interval_half_zolotarev(matvec!, n, zp; ...)` *(experimental)*

Filter construction:
- `ZolotarevParams(nodes, G; ...)`

#### When to use
- same setting as FEAST, but you want better performance
- want fewer iterations / poles
- the interval is wide or spectrally challenging

#### Why it works better
- near-optimal rational approximation
- sharper separation of wanted vs unwanted eigenvalues
- fewer shifted solves for the same accuracy

For real symmetric matrices, the half-plane version is usually the best overall solver in this repository.

---

### 3. Chebyshev Subspace Iteration

This method uses polynomial filtering instead of rational filtering. A Chebyshev polynomial \(p(A)\) is constructed such that

\[
Y = p(A)Q
\]

amplifies components inside the interval and damps the rest.

#### High-level API

- `chebyshev_poly_real_symm_sparse(A, emin, emax; ...)`
- `chebyshev_poly_real_symm_sparse(matvec!, n, emin, emax; ...)`

#### When to use
- only have matrix-vector products
- want to avoid any linear solves (LU)
- are working in a matrix-free setting

#### Limitations
- requires higher polynomial degree for sharp filters
- less efficient for narrow intervals (check tests)

> In practice, POLFED (below) is often a better polynomial-based alternative.

---

### 4. POLFED

POLFED is a polynomial filtering eigensolver producing more effective filters than standard Chebyshev subspace iteration.

#### High-level API

- `polfed(...)` (matrix and matrix-free variants)

#### When to use
- want an LU free interval eigensolver
- need many eigenvalues (1000) efficiently

#### Why it works
- avoids shifted solves entirely
- scales better for large problems than chebyshev subspace iteration

---

## Choosing the Right Method

### If you have sparse matrices and can factorize:
Use Zolotarev FEAST (preferred) 
- fastest convergence  
- best interval selectivity  
- ideal for many eigenvalues  

Use standard FEAST
- want a simpler / classical formulation  
- or for comparison / validation  

---

### If you cannot use shifted solves:
Use POLFED (preferred)
- strongest solve-free method  
- better filtering than Chebyshev subspace iteration 

---

### If the problem is symmetric:
Always use half-contour / half-plane variants 
- reduces cost by at least ~2× (QR workspaces for real are cheaper and faster etc.)
- improves numerical behavior  

---

## Practical Rule of Thumb

- Best overall method when you can make a concrete matrix and afford LU:
  Zolotarev FEAST (half-plane for symmetric problems)

- No factorizations available:
  POLFED

---

## Dependencies

- `LinearAlgebra`
- `SparseArrays`
- `LinearSolve.jl`
- `Krylov.jl`
- `QuadGK.jl`

Optional:
- `MKL.jl`
- `MKLSparse.jl`
