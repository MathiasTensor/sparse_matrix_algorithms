# ==============================================================================
# FEAST 
# ==============================================================================
#
# This file implements FEAST-type algorithms for computing interior eigenpairs
# of large sparse (or matrix-free) operators using contour integration and
# subspace iteration.
#
# ------------------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------------------
#
# We solve the eigenvalue problem
#
#     A x = λ x
#
# for eigenvalues λ inside a target interval [emin, emax].
#
# FEAST approximates the spectral projector
#
#     P = (1 / (2π i)) ∮_Γ (zI - A)^{-1} dz
#
# using numerical quadrature over a complex contour Γ (ellipse), yielding:
#
#     P ≈ Σ_k w_k (z_k I - A)^{-1}
#
# Applying this projector to a subspace and performing Rayleigh–Ritz yields
# eigenpairs in the desired interval.
#
# ------------------------------------------------------------------------------
# Implemented Variants
# ------------------------------------------------------------------------------
#
# 1. general_feast_sparse_interval
#    - Full contour FEAST (complex arithmetic)
#    - Works for general real matrices
#
# 2. sym_feast_sparse_interval_half
#    - Half-contour FEAST for real symmetric matrices
#    - Exploits conjugate symmetry:
#         P Q ≈ 2 Re Σ w_k (z_k I - A)^{-1} Q
#    - ~2× fewer shifted solves, real arithmetic in RR
#
# 3. *_distributed variants
#    - Lower memory usage (no stored factorizations)
#    - Factorizations recomputed per iteration
#    - Slower but scalable to larger problems
#
# 4. MatrixFreeRealSymOp
#    - Matrix-free operator interface
#    - Supports vector and block matvecs
#
# ------------------------------------------------------------------------------
# Algorithm (per FEAST iteration)
# ------------------------------------------------------------------------------
#
# Given subspace Q ∈ ℂ^{n×m0}:
#
#   1. Q ← orth(Q)
#   2. Y = Σ_k w_k (z_k I - A)^{-1} Q        (dominant cost)
#   3. Q ← orth(Y)
#   4. Rayleigh–Ritz:
#          Aq = Qᴴ A Q
#          Aq u = λ u
#          X = Q u
#   5. Residual filtering and convergence check
#
# Key Parameters
# ------------------------------------------------------------------------------
#
# - m0       : subspace size (must exceed # eigenvalues in interval)
# - nodes    : number of contour nodes (tradeoff: accuracy vs cost)
# - eta      : ellipse height parameter (conditioning vs selectivity)
# - backend  : linear solver (:lu or :ls via LinearSolve.jl)
# - blockrhs : RHS block size (distributed variants)
#
# Reference: #   FEAST Eigensolver for Nonlinear Eigenvalue Problems, Gavin B., Międlar, A, Pollizi E. https://arxiv.org/abs/1801.09794
#
# BROUGHT TO YOU BY THE POWER OF CHATGPT-5.2
# EDITING: Orel 10/3/2026
# * If editing file/want to add things, give your surname and date in the comment above, and briefly describe what you changed
# ==============================================================================

using LinearAlgebra
using SparseArrays
using Random
using Printf
using BenchmarkTools
using LinearSolve
using ProgressMeter
using Statistics

try
    @eval using MKL
catch
    @warn "MKL.jl not installed/available"
end
try
    @eval using MKLSparse
catch
    @warn "MKLSparse.jl not installed/available"
end

"""
    MatrixFreeRealSymOp{F,G}

Matrix-free real symmetric operator. Represents a real symmetric linear map A on ℝⁿ without explicitly storing a full
matrix. If `matblock!` is supplied, block applications can be more efficient
than repeated columnwise `matvec!` calls.

Fields
------
- `n::Int`: operator dimension.
- `matvec!::F`: in-place action `y ← A*x`.
- `matblock!::Union{Nothing,G}`: optional in-place block action `Y ← A*X`.
"""
struct MatrixFreeRealSymOp{F,G,H}
    n::Int
    matvec!::F
    matblock!::Union{Nothing,G}
end
"""
    MatrixFreeRealSymOp(n,matvec!)

Construct a matrix-free operator with only vector action.
"""
MatrixFreeRealSymOp(n::Int,matvec!::F) where {F}=MatrixFreeRealSymOp{F,Nothing,Nothing}(n,matvec!,nothing)
"""
    MatrixFreeRealSymOp(n,matvec!,matblock!)

Construct a matrix-free operator with both vector and block actions.
"""
MatrixFreeRealSymOp(n::Int,matvec!::F,matblock!::G) where {F,G}=MatrixFreeRealSymOp{F,G,Nothing}(n,matvec!,matblock!)

# helper for size of the operator as a matrix
Base.size(A::MatrixFreeRealSymOp)=(A.n,A.n)
Base.size(A::MatrixFreeRealSymOp,d::Int)=(d==1) || (d==2) ? A.n : 1
"""
    LinearAlgebra.mul!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})

Matrix-free matrix-vector product for a real symmetric operator `A` represented by a `MatrixFreeRealSymOp`. Computes `y <- A*x` using the user-provided `matvec!` function.
"""
@inline function LinearAlgebra.mul!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})
    A.matvec!(y,x)
    return y
end
"""
    apply_A_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})

Block matrix-free application of a real symmetric operator `A` to a block of vectors `X`, storing the result in `Y`. Computes `Y[:,j] <- A*X[:,j]` for each column `j` of `X`. 
"""
@inline function apply_A_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})
    if !isnothing(A.matblock!)
        A.matblock!(Y,X)
        return Y
    else
        n,m=size(X)
        @inbounds for j in 1:m
            mul!(view(Y,:,j),A,view(X,:,j))
        end
        return Y
    end
end

"""
    struct DiagPtr{Ti<:Integer}
        p::Vector{Ti}
    end

A small helper that stores, for each column/row j, the index `t` into `A.nzval`
where the diagonal entry A[j,j] is stored (in CSC format), i.e.

    A.nzval[p[j]] == A[j,j]

If the diagonal entry is structurally missing, p[j] = 0.

Why we need this:
- In FEAST we repeatedly form shifted matrices (zI - A) for many complex shifts z.
- We want to update ONLY the diagonal in-place, keeping the same sparse pattern.
- So we need O(n) access to diagonal locations in the CSC storage arrays.
"""
struct DiagPtr{Ti<:Integer}
    p::Vector{Ti} # p[j] = index into nzval of A[j,j], or 0 if missing
end

"""
    diagptr(A::SparseMatrixCSC{T,Ti}) -> DiagPtr{Ti}

Scan each CSC column j of A, locate the diagonal entry (row == j),
and store its nzval index in p[j]. If missing, p[j]=0.

Cost: O(nnz) worst-case, but done once per A.
"""
function diagptr(A::SparseMatrixCSC{T,Ti}) where {T,Ti<:Integer}
    n=size(A,1)
    p=zeros(Ti,n)
    @inbounds for j in 1:n
        for t in A.colptr[j]:(A.colptr[j+1]-1)
            if A.rowval[t]==j
                p[j]=Ti(t)
                break
            end
        end
    end
    return DiagPtr{Ti}(p)
end

"""
    ensure_diagonal(A::SparseMatrixCSC{Float64,Int}) -> SparseMatrixCSC{Float64,Int}

Guarantee that every diagonal entry A[j,j] exists structurally in the sparse
pattern (even if its numerical value is 0).

Why:
- If a diagonal entry is missing from the sparse structure, we cannot shift it
  without changing sparsity (which would break LU reuse and be expensive).
- Adding `spdiagm(0 => zeros(n))` inserts missing diagonal structural zeros.

Note:
- This changes the sparsity structure (adds diagonal zeros), but only once.
"""
function ensure_diagonal(A::SparseMatrixCSC{Float64,Int})
    n=size(A,1)
    dp=diagptr(A)
    if any(dp.p.==0)
        A=A+spdiagm(0=>zeros(n))
    end
    return A
end

"""
    shift_diag!(Az, A0, dp, z)

Given:
- `A0` : real sparse matrix (Float64) with an explicit diagonal
- `Az` : complex sparse matrix with IDENTICAL sparsity pattern as A0
- `dp` : diagonal pointer map for A0 (and Az, since sparsity matches)
- `z`  : complex shift

Overwrite ONLY the diagonal entries of Az to represent:

    Az := A0 - z I

i.e. Az[j,j] = A0[j,j] - z, while leaving off-diagonal entries unchanged.

This is used when assembling (zI - A) efficiently:
- We start from Az = A (complex copy)
- We shift its diagonal to (A - zI)
- Then we negate all values to obtain (zI - A)
"""
@inline function shift_diag!(Az::SparseMatrixCSC{ComplexF64,Ti},A0::SparseMatrixCSC{Float64,Ti},dp::DiagPtr{Ti},z::ComplexF64) where {Ti<:Integer}
    @inbounds for j in 1:length(dp.p)
        t=dp.p[j]
        t==0 && throw(ArgumentError("missing diagonal entry at ($j,$j); call ensure_diagonal(A)"))
        Az.nzval[t]=ComplexF64(A0.nzval[t])-z
    end
    return nothing
end

"""
    residuals!(res, A, X, λ, AX)

Compute absolute residual norms for approximate eigenpairs (λ_j, x_j):

    res[j] = ||A*x_j - λ_j*x_j||_2

Inputs:
- A  : sparse real matrix
- X  : n×m matrix of (complex) approximate eigenvectors (columns)
- λ  : length-m vector of (complex) approximate eigenvalues
- AX : workspace n×m to store A*X

Notes:
- For symmetric real A and a good FEAST subspace, λ should be ~ real.
- Absolute residuals are fine for scaled tests; for production, a relative
  residual is often more meaningful:
      ||A x - λ x|| / (||A x|| + |λ| ||x||).
"""
function residuals!(res::Vector{Float64},A::SparseMatrixCSC{Float64,Int},X::Matrix{T},λ::Vector{T},AX::Matrix{T}) where T<:Union{Float64,ComplexF64}
    mul!(AX,A,X)
    n,m=size(X)
    @inbounds for j in 1:m
        lj=λ[j]
        s=0.0
        for i in 1:n
            v=AX[i,j]-lj*X[i,j]
            s+=real(v*conj(v))
        end
        res[j]=sqrt(s)
    end
    return nothing
end

"""
    thin_orth!(Qc, Qtmp, Z)

Replace Qc by an orthonormal basis of its column space:

    Qc ← orth(Qc)

Parameters:
- Qc: n×m0 matrix to be orthonormalized in-place (columns replaced by orthonormal basis)
- Qtmp, Z: n×m0 workspace matrices (can be reused across calls)


and ensure the result has size n×m0.

Implementation:
- Compute QR factorization: Qc = Q R.
- Materialize Q as a dense matrix and keep its first m0 columns.

Why we do this:
- FEAST is a subspace method. It requires Rayleigh–Ritz in an ORTHONORMAL basis.
- After projector application, columns become correlated / ill-conditioned.
- Orthonormalizing stabilizes the subspace iteration and ensures that the small
  projected matrix Aq = Q'*A*Q is (numerically) Hermitian for symmetric A.
"""
function thin_orth!(Qc::Matrix{Float64},Qtmp::Matrix{Float64},τ::Vector{Float64})
    n,m=size(Qc) # dims of Qc
    k=min(n,m) # smallest dimension (m0)
    LAPACK.geqrf!(Qc,τ) # QR factorization of Qc with reflectors τ
    copyto!(@view(Qtmp[:,1:k]),@view(Qc[:,1:k])) # get the m0 columns of Qc (inplace QR) and copy to temp
    LAPACK.orgqr!(@view(Qtmp[:,1:k]),τ,k) # get Q (orthogonal) matrix from Qtmp above
    copyto!(Qc,@view(Qtmp[:,1:k])) # copy to Qc the mo columns of the matrix Q (in Qc=Q*R)
    return nothing
end
# there is no LAPACK.ungqr! exposed in LinearAlgebra.jl
function thin_orth!(Qc::Matrix{ComplexF64},Qtmp::Matrix{ComplexF64},Z::Matrix{ComplexF64})
    F=qr!(Qc)
    mul!(Qtmp,F.Q,Z)
    copyto!(Qc,Qtmp)
    return nothing
end
# workspaces for not allocating thin_orth! each iter in max_iter loop in general_feast_sparse_interval
function make_thin_orth_workspace(::Type{ComplexF64},n::Int,m0::Int)
    Qtmp=Matrix{ComplexF64}(undef,n,m0)
    Z=zeros(ComplexF64,n,m0)
    @inbounds for j in 1:m0
        Z[j,j]=1.0+0.0im
    end
    return Qtmp,Z
end
function make_thin_orth_workspace(::Type{Float64},n::Int,m0::Int)
    Qtmp=Matrix{Float64}(undef,n,m0)
    τ=Vector{Float64}(undef,min(n,m0))
    return Qtmp,τ
end

"""
    ellipse_nodes_weights(emin, emax; nodes=64, eta=0.6) -> (z, w)

Construct contour nodes `z[k]` and quadrature weights `w[k]` that approximate
the spectral projector integral:

    P = (1/(2π i)) ∮_Γ (zI - A)^{-1} dz
      ≈ Σ_{k=1..nodes} w[k] * (z[k] I - A)^{-1}.

Contour Γ: an ellipse enclosing the real interval [emin, emax], lifted off the
real axis to avoid near-singular shifted solves:
    z(θ) = c + a cosθ + i b sinθ
    c = (emin+emax)/2
    a = (emax-emin)/2
    b = eta*a   (vertical semi-axis)

Trapezoidal rule with θ_k = (k-1)Δθ, Δθ = 2π/nodes:

    z_k = z(θ_k)
    dz  = z'(θ_k) Δθ
    w_k = (1/(2π i)) dz

The factor (1/(2π i)) is built into w_k so you can write directly:
    Y = Σ w_k (z_k I - A)^{-1} Q.
"""
function ellipse_nodes_weights(emin::Float64,emax::Float64;nodes::Int=64,eta::Float64=0.6)
    nodes>0 || throw(ArgumentError("nodes must be positive"))
    c=0.5*(emin+emax)
    a=0.5*(emax-emin)
    a>0 || throw(ArgumentError("need emin<emax"))
    b=eta*a
    Δθ=2*pi/nodes
    z=Vector{ComplexF64}(undef,nodes)
    w=Vector{ComplexF64}(undef,nodes)
    @inbounds for k in 1:nodes
        θ=(k-0.5)*Δθ  
        ct=cos(θ); st=sin(θ)
        z[k]=c+a*ct+im*b*st
        dz=(-a*st+im*b*ct)*Δθ
        w[k]=dz/(2*pi*im)
    end
    return z,w
end

"""
    ellipse_nodes_weights_half(emin, emax; nodes=64, eta=0.6) -> (z_half, w_half)

Construct contour nodes and weights for the half-ellipse that encloses [emin, emax] in the upper half-plane. Used in the real symmetric case to exploit conjugate symmetry and reduce the number of shifted solves by half.
"""
function ellipse_nodes_weights_half(emin::Float64,emax::Float64;nodes::Int=64,eta::Float64=0.6)
    iseven(nodes) || throw(ArgumentError("half-node requires even nodes"))
    z,w=ellipse_nodes_weights(emin,emax;nodes=nodes,eta=eta)
    nh=nodes÷2
    return z[1:nh],w[1:nh] # midpoint makes imag(z)>0 for first half
end

"""
    general_feast_sparse_interval(Ain; emin, emax, m0=200, nodes=64, eta=0.6, maxiter=8, tol=1e-10, res_gate=1e-6, debug=false, is_hermitian=true) -> (λsel::Vector{Float64}, Xsel::Matrix{ComplexF64}, ressel::Vector{Float64})

Compute eigenpairs of a real symmetric sparse matrix A with eigenvalues in the
target interval [emin, emax].
---------------------------------------
Algorithm (mathematical steps)
---------------------------------------

Let Γ be the ellipse enclosing [emin, emax] (lifted off the real axis).
Define the spectral projector onto eigenvectors inside Γ:

    P = (1/(2π i)) ∮_Γ (zI - A)^{-1} dz.

Approximate P by trapezoidal quadrature:

    P ≈ Σ_{k=1..nodes} w_k (z_k I - A)^{-1}.

Subspace iteration (rational filtering):

1) Initialize Q ∈ ℂ^{n×m0} random.
2) For it = 1..maxiter:
    a) Orthonormalize:      Q ← orth(Q).
    b) Apply projector:     Y = P Q ≈ Σ w_k (z_k I - A)^{-1} Q.
    c) Orthonormalize:      Q ← orth(Y).
    d) Rayleigh–Ritz:
           Aq = Q* A Q  (m0×m0)
           Solve Aq u = λ u (small dense problem)
           Lift x = Q u.
    e) Compute residuals r_j = ||A x_j - λ_j x_j||.
    f) Count eigenpairs inside interval (and optionally below res_gate), stop if max residual < tol.
    g) Finally, return the subset with λ in [emin, emax] and residual below a gate.

---------------------------------------
Parameters
---------------------------------------
- m0:
    Size of the search subspace. Must exceed the number of eigenvalues inside
    [emin, emax] (or Γ), plus padding.
    If m0 is too small you will miss eigenvalues.
- nodes:
    Number of quadrature nodes on Γ. Larger nodes => better approximation of P,
    but more factorizations and solves.
- eta:
    Imaginary lift of ellipse (b = eta*a). Too small => (zI-A) close to singular
    near real spectrum; too large => filter becomes less selective.
- tol:
    Convergence criterion based on residuals of accepted eigenpairs.
- res_gate:
    A reporting/selection threshold used for counting inside.
    For debugging, you may set this large or ignore it; for production,
    this prevents counting Ritz junk.
- is_hermitian:
    If true, assume Aq is Hermitian and use Hermitian eigensolver.
    This is correct when Q is orthonormal and A is symmetric.
- debug:
    If true, print iteration info (number of inside eigenvalues, max residual).
- show_progress:
    If true, display progress bars for factorization and iteration loops.
- backend:
    :lu (default) uses direct LU factorization for shifted solves.
    :ls uses an iterative linear solver (e.g. GMRES) for shifted solves.
- ls_alg:
    If backend=:ls, you can specify the iterative solver algorithm (e.g. MKLPardisoFactorization()) from LinearSolve.jl; if not provided, the default is used
- two_hit:
    If true (default), require two consecutive iterations with maxres<tol to declare convergence, for robustness against transient small residuals.

Returns:
- λsel: selected eigenvalues (Float64, sorted)
- Xsel: corresponding eigenvectors (columns) in ℂ^{n×m}
- ressel: residual norms for those selected eigenpairs
"""
function general_feast_sparse_interval(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,nodes::Int=64,eta::Float64=0.6,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,is_hermitian::Bool=true,show_progress::Bool=false,backend::Symbol=:lu,ls_alg=nothing,two_hit::Bool=true)
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    nodes>=8 || throw(ArgumentError("nodes too small; use 32/64+ for robustness"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    A=ensure_diagonal(Ain)
    z,w=ellipse_nodes_weights(emin,emax;nodes=nodes,eta=eta)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Qc=ComplexF64.(randn(n,m0))
    Y=Matrix{ComplexF64}(undef,n,m0)
    Tm=Matrix{ComplexF64}(undef,n,m0)
    R=Matrix{ComplexF64}(undef,n,m0)
    Aq=Matrix{ComplexF64}(undef,m0,m0)
    Xq=Matrix{ComplexF64}(undef,m0,m0)
    λ=Vector{ComplexF64}(undef,m0)
    X=Matrix{ComplexF64}(undef,n,m0)
    AX=Matrix{ComplexF64}(undef,n,m0)
    res=Vector{Float64}(undef,m0);inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(ComplexF64,n,m0)
    Ffact=backend===:lu ? Vector{Any}(undef,nodes) : nothing
    lins=backend===:ls ? Vector{Any}(undef,nodes) : nothing
    b0=backend===:ls ? zeros(ComplexF64,n,m0) : nothing
    u0=backend===:ls ? similar(b0) : nothing
    # This part prepares the P = Σ w_k (z_k I - A)^{-1} operator by factorizing (z_k I - A) for each node k. This is not 
    # the most expensive part, but can be significant for large nodes.
    p_fact=show_progress ? Progress(nodes;desc="FEAST: factorizations") : nothing
    @inbounds for k in 1:nodes
        copyto!(Az.nzval,base_nz)
        shift_diag!(Az,A,dp,z[k])
        @inbounds for t in eachindex(Az.nzval)
            Az.nzval[t]=-Az.nzval[t] # (zI-A)
        end
        if backend===:lu
            Ffact[k]=lu(Az)
        elseif backend===:ls
            Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
            prob=LinearSolve.LinearProblem(Ak,b0;u0=u0)
            lins[k]=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
        else
            throw(ArgumentError("backend must be :lu or :ls"))
        end
        show_progress && next!(p_fact)
    end
    vY=vec(Y)
    # the reall bottleneck, as it requires maxiter*nodes solves; the factorization above is a one-time cost while solve! 
    # is really expensive and dominates the runtime for large problems.
    prev_ok=false
    for it in 1:maxiter
        p_it=show_progress ? Progress(nodes;desc="FEAST: subspace iteration ($(it))") : nothing
        fill!(Y,0.0+0.0im)
        @inbounds for k in 1:nodes
            if backend===:lu
                ldiv!(Tm,Ffact[k],Qc) # solve (zI-A) Tm = Qc
                LinearAlgebra.BLAS.axpy!(w[k],vec(Tm),vY) # Y += w[k] * Tm (accumulate into Y)
                show_progress && next!(p_it)
            else
                lins[k].b=Qc
                sol=LinearSolve.solve!(lins[k]) # solve (zI-A) Tm = Qc using iterative solver
                LinearAlgebra.BLAS.axpy!(w[k],vec(sol.u),vY) # Y += w[k] * Tm (accumulate into Y)
                show_progress && next!(p_it)
            end
        end
        Qc.=Y
        thin_orth!(Qc,Qtmp,Z) # relatively cheap since nodes loop dominates
        mul!(R,A,Qc) # BLAS gemm: O(n m0^2)
        mul!(Aq,adjoint(Qc),R) # BLAS gemm: O(n m0^2)
        if is_hermitian
            E=eigen!(Hermitian(Aq)) # O(m0^3) but m0 is small!!!
            λ.=ComplexF64.(E.values)
            Xq.=E.vectors
        else
            E=eigen(Aq) # O(m0^3) but m0 is small!!!
            λ.=E.values
            Xq.=E.vectors
        end
        mul!(X,Qc,Xq) # O(n m0^2)
        residuals!(res,A,X,λ,AX) # O(n m0^2) but dominated by solves
        # each solve is O(n^2) for direct LU, or depends on the iterative solver and convergence (check LinearSolve.jl docs)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(real(λ[j])≥emin && real(λ[j])≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("inside=%d  maxres=%.6e\n",count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (real(λ[j])≥emin && real(λ[j])≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{ComplexF64}(undef,n,0),Float64[]
    λsel=real.(λ[idx]);p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    general_feast_sparse_interval_distributed(Ain; emin, emax, m0=200, nodes=64, eta=0.6,
                                      maxiter=8, tol=1e-10, res_gate=1e-6,
                                      debug=false, is_hermitian=true,
                                      backend=:lu, ls_alg=nothing, blockrhs=0, two_hit=true, show_progress=false)

Low-memory FEAST implementation for computing eigenpairs of a large sparse real matrix

    A x = λ x

with eigenvalues in the interval `[emin, emax]`.

Algorithmically this is identical to `general_feast_sparse_interval`:

    P = (1/(2π i)) ∮_Γ (zI - A)⁻¹ dz
      ≈ Σ_k w_k (z_k I - A)⁻¹

and the FEAST iteration is

    Q ← orth(Q)
    Y = Σ_k w_k (z_k I - A)⁻¹ Q
    Q ← orth(Y)
    Aq = Qᴴ A Q
    solve Aq u = λ u
    x = Q u

Returns:
    λsel  – eigenvalues in `[emin, emax]`
    Xsel  – corresponding eigenvectors
    ressel – residual norms
"""
function general_feast_sparse_interval_distributed(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,nodes::Int=64,eta::Float64=0.6,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,is_hermitian::Bool=true,backend::Symbol=:lu,ls_alg=nothing,blockrhs::Int=0,show_progress::Bool=false,two_hit::Bool=true)
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    nodes>=8 || throw(ArgumentError("nodes too small"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)
    A=ensure_diagonal(Ain)
    z,w=ellipse_nodes_weights(emin,emax;nodes=nodes,eta=eta)
    # Sparse shifted operator template: reuse (colptr,rowval); only nzval changes with z_k.
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    
    # unavoidable O(n*m0) memory
    Qc=ComplexF64.(randn(n,m0))
    Y=Matrix{ComplexF64}(undef,n,m0)
    Tm=Matrix{ComplexF64}(undef,n,m0)
    R=Matrix{ComplexF64}(undef,n,m0)
    Aq=Matrix{ComplexF64}(undef,m0,m0)
    Xq=Matrix{ComplexF64}(undef,m0,m0)
    λ=Vector{ComplexF64}(undef,m0)
    X=Matrix{ComplexF64}(undef,n,m0)
    AX=Matrix{ComplexF64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(ComplexF64,n,m0)
    blockrhs<=0 && (blockrhs=m0)
    blockrhs=min(blockrhs,m0)
    Bblk=Matrix{ComplexF64}(undef,n,blockrhs)
    Ublk=Matrix{ComplexF64}(undef,n,blockrhs)

    vY=vec(Y)
    # FEAST: spectral projector P = (1/2πi)∮(zI-A)^{-1}dz ≈ Σ_k w_k (z_k I - A)^{-1} 
    prev_ok=false
    for it in 1:maxiter
        fill!(Y,0.0+0.0im)
        # Y = Σ_k w_k (z_k I - A)^{-1} Qc
        progress_k=show_progress ? Progress(nodes;desc="FEAST: node iteration, i=($(it)) from maxiter=$(maxiter)") : nothing
        @inbounds for k in 1:nodes
            copyto!(Az.nzval,base_nz)
            shift_diag!(Az,A,dp,z[k])
            @inbounds for t in eachindex(Az.nzval)
                Az.nzval[t]=-Az.nzval[t] # (zI-A)
            end 
            wk=w[k]
            if backend===:lu
                F=lu(Az)
                if blockrhs==m0
                    ldiv!(Tm,F,Qc)
                    LinearAlgebra.BLAS.axpy!(wk,vec(Tm),vY)
                else
                    j0=1
                    while j0<=m0
                        bs=min(blockrhs,m0-j0+1)
                        @views copyto!(Bblk[:,1:bs],Qc[:,j0:j0+bs-1])
                        bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                        ldiv!(Ublk,F,Bblk)
                        @views Y[:,j0:j0+bs-1].+=wk.*Ublk[:,1:bs]
                        j0+=bs
                    end
                end
            elseif backend===:ls
                Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
                if blockrhs==m0
                    prob=LinearSolve.LinearProblem(Ak,Qc;u0=Tm)
                    lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                    sol=LinearSolve.solve!(lin)
                    LinearAlgebra.BLAS.axpy!(wk,vec(sol.u),vY)
                else
                    j0=1
                    while j0<=m0
                        bs=min(blockrhs,m0-j0+1)
                        @views copyto!(Bblk[:,1:bs],Qc[:,j0:j0+bs-1])
                        bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                        prob=LinearSolve.LinearProblem(Ak,Bblk;u0=Ublk)
                        lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                        sol=LinearSolve.solve!(lin)
                        @views Y[:,j0:j0+bs-1].+=wk.*sol.u[:,1:bs]
                        j0+=bs
                    end
                end
            else
                throw(ArgumentError("backend must be :lu or :ls"))
            end
            show_progress && next!(progress_k)
        end
        # Next subspace
        Qc.=Y
        thin_orth!(Qc,Qtmp,Z) # cost is O(n m0^2)
        # Rayleigh–Ritz: Aq = Qᴴ A Q (m0×m0), eigen solve is O(m0^3) and tiny vs sparse solves.
        mul!(R,A,Qc) # BLAS gemm: O(n m0^2)
        mul!(Aq,adjoint(Qc),R) # BLAS gemm: O(n m0^2)
        if is_hermitian
            E=eigen!(Hermitian(Aq)) # O(m0^3) but m0 is small!!!
            λ.=ComplexF64.(E.values) 
            Xq.=E.vectors
        else
            E=eigen(Aq) # O(m0^3) but m0 is small!!!
            λ.=E.values
            Xq.=E.vectors
        end
        # Lift eigenvectors + residual filter
        mul!(X,Qc,Xq) # O(n m0^2)
        residuals!(res,A,X,λ,AX) # O(n m0^2) 
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(real(λ[j])≥emin && real(λ[j])≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (real(λ[j])≥emin && real(λ[j])≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{ComplexF64}(undef,n,0),Float64[]
    λsel=real.(λ[idx]);p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    sym_feast_sparse_interval_half(Ain; emin, emax, m0=200, nodes=64, eta=0.6,
                                   maxiter=8, tol=1e-10, res_gate=1e-6,
                                   debug=false, show_progress=false,
                                   backend=:lu, ls_alg=nothing, two_hit=true)

FEAST eigensolver for a real symmetric sparse matrix

    A x = λ x

that computes eigenpairs with eigenvalues in the interval `[emin, emax]`.

This version uses a half-contour quadrature to accelerate the spectral
projector evaluation. For real matrices and real FEAST subspaces `Q`,
contributions from conjugate contour nodes satisfy

    w (zI - A)⁻¹ Q + w̄ (z̄I - A)⁻¹ Q = 2 Re[w (zI - A)⁻¹ Q].

Therefore the projector can be evaluated using only the nodes in the
upper half-plane, reducing the number of shifted linear solves by
approximately a factor of two.

The Rayleigh–Ritz step is performed in a real subspace and the projected
matrix is solved using a symmetric eigensolver.

# Parameters
- `m0` : dimension of the FEAST search subspace (must exceed the number of eigenvalues in the interval).
- `nodes` : number of quadrature nodes on the full contour (only `nodes/2` solves are performed).
- `eta` : imaginary lift of the ellipse contour (`b = eta*a`).
- `maxiter` : maximum FEAST iterations.
- `tol` : convergence tolerance based on residual norms.
- `res_gate` : residual threshold used when selecting eigenpairs inside the interval.
- `backend` : solver for shifted systems (`:lu` or `:ls` via LinearSolve.jl).
- `ls_alg` : optional LinearSolve algorithm.
- `two_hit` : if true, require two consecutive iterations with maxres<tol to declare convergence, for robustness against transient small residuals.

# Returns
- `λsel::Vector{Float64}` : eigenvalues in `[emin, emax]` (sorted).
- `Xsel::Matrix{Float64}` : corresponding eigenvectors.
- `ressel::Vector{Float64}` : residual norms.
"""
function sym_feast_sparse_interval_half(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,nodes::Int=64,eta::Float64=0.6,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,show_progress::Bool=false,backend::Symbol=:lu,ls_alg=nothing,two_hit::Bool=true)

    issymmetric(Ain) || throw(ArgumentError("half-contour requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    nodes>=8 || throw(ArgumentError("nodes too small"))
    iseven(nodes) || throw(ArgumentError("half-node requires even nodes"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))

    A=ensure_diagonal(Ain)
    zh,wh=ellipse_nodes_weights_half(emin,emax;nodes=nodes,eta=eta)
    nh=length(zh)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Q=randn(n,m0)
    Y=zeros(Float64,n,m0)
    # Complex solve workspaces
    Bc=Matrix{ComplexF64}(undef,n,m0) 
    Tm=Matrix{ComplexF64}(undef,n,m0) 
    vTm=vec(Tm)
    R=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)
    Ffact=backend===:lu ? Vector{Any}(undef,nh) : nothing
    lins=backend===:ls ? Vector{Any}(undef,nh) : nothing
    b0=backend===:ls ? zeros(ComplexF64,n,m0) : nothing
    u0=backend===:ls ? similar(b0) : nothing
    p_fact=show_progress ? Progress(nh;desc="FEAST(sym,half): factorizations") : nothing
    @inbounds for k in 1:nh
        copyto!(Az.nzval,base_nz)
        shift_diag!(Az,A,dp,zh[k])
        @inbounds for t in eachindex(Az.nzval)
            Az.nzval[t]=-Az.nzval[t] # (zI-A)
        end
        if backend===:lu
            Ffact[k]=lu(Az)
        elseif backend===:ls
            Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
            prob=LinearSolve.LinearProblem(Ak,b0;u0=u0)
            lins[k]=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
        else
            throw(ArgumentError("backend must be :lu or :ls"))
        end
        show_progress && next!(p_fact)
    end
    prev_ok=false
    for it in 1:maxiter
        p_it=show_progress ? Progress(nh;desc="FEAST(sym,half): iter $(it)") : nothing
        fill!(Y,0.0)
        @views for j in 1:m0
            @inbounds for i in 1:n
                Bc[i,j]=Q[i,j]
            end
        end
        @inbounds for k in 1:nh
            if backend===:lu
                ldiv!(Tm,Ffact[k],Bc)
            else
                lins[k].b=Bc
                sol=LinearSolve.solve!(lins[k])
                copyto!(Tm,sol.u)
            end
            wk=wh[k]
            @inbounds for idx in eachindex(vTm)
                Y[idx]+=2.0*real(wk*vTm[idx])   # half-node accumulation
            end
            show_progress && next!(p_it)
        end
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        mul!(R,A,Q)
        mul!(Aq,transpose(Q),R)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx];p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end

"""
    sym_feast_sparse_interval_half_distributed(Ain; emin, emax, m0=200, nodes=64, eta=0.6, maxiter=8, tol=1e-10, res_gate=1e-6, debug=false, backend=:lu, ls_alg=nothing, blockrhs=0, show_progress=false, two_hit=true)

Low-memory FEAST implementation for computing eigenpairs of a real
symmetric sparse matrix

    A x = λ x

with eigenvalues in `[emin, emax]`.

This routine uses the same half-contour spectral projector
as `sym_feast_sparse_interval_half`, exploiting conjugate symmetry of
the contour quadrature to evaluate

    P Q ≈ 2 Σ Re[w_k (z_k I - A)⁻¹ Q]

using only the contour nodes in the upper half-plane.

Unlike the pre-factorized version, this implementation **does not store
factorizations for all contour nodes**. Each shifted system `(zI-A)` is
assembled and solved on the fly, reducing memory usage for large
problems.

Right-hand sides may be processed in blocks controlled by `blockrhs`.

# Parameters
- `m0` : FEAST search subspace dimension.
- `nodes` : number of quadrature nodes on the full contour.
- `eta` : ellipse lift parameter.
- `blockrhs` : number of right-hand sides solved simultaneously.
- `backend` : linear solver (`:lu` or `:ls`).
- `ls_alg` : optional LinearSolve algorithm.
- `two_hit` : if true, require two consecutive iterations with maxres<tol to declare convergence.

# Returns
- `λsel::Vector{Float64}` : eigenvalues in `[emin, emax]`.
- `Xsel::Matrix{Float64}` : corresponding eigenvectors.
- `ressel::Vector{Float64}` : residual norms.
"""
function sym_feast_sparse_interval_half_distributed(Ain::SparseMatrixCSC{Float64,Int};emin::Float64,emax::Float64,m0::Int=200,nodes::Int=64,eta::Float64=0.6,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,debug::Bool=false,backend::Symbol=:lu,ls_alg=nothing,blockrhs::Int=0,show_progress::Bool=false,two_hit::Bool=true)

    issymmetric(Ain) || throw(ArgumentError("half-contour requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    nodes>=8 || throw(ArgumentError("nodes too small"))
    iseven(nodes) || throw(ArgumentError("half-node requires even nodes"))
    n=size(Ain,1);size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)

    A=ensure_diagonal(Ain)
    zh,wh=ellipse_nodes_weights_half(emin,emax;nodes=nodes,eta=eta)
    nh=length(zh)
    dp=diagptr(A)
    base_nz=ComplexF64.(A.nzval)
    Az=SparseMatrixCSC{ComplexF64,Int}(A.m,A.n,copy(A.colptr),copy(A.rowval),similar(base_nz))
    Q=randn(n,m0)
    Y=zeros(Float64,n,m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)
    R=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    # block RHS controls memory/solve shape
    blockrhs<=0 && (blockrhs=m0)
    blockrhs=min(blockrhs,m0)
    # complex block workspaces (only these are complex)
    Bblk=Matrix{ComplexF64}(undef,n,blockrhs)
    Ublk=Matrix{ComplexF64}(undef,n,blockrhs)
    prev_ok=false
    for it in 1:maxiter
        fill!(Y,0.0)
        progress_k=show_progress ? Progress(nh;desc="FEAST(sym,half,dist): iter $(it)") : nothing
        @inbounds for k in 1:nh
            copyto!(Az.nzval,base_nz)
            shift_diag!(Az,A,dp,zh[k])
            @inbounds for t in eachindex(Az.nzval)
                Az.nzval[t]=-Az.nzval[t] # (zI-A)
            end
            wk=wh[k]
            if backend===:lu
                F=lu(Az)
                j0=1
                while j0<=m0
                    bs=min(blockrhs,m0-j0+1)
                    @views Bblk[:,1:bs].=Q[:,j0:j0+bs-1]
                    bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                    ldiv!(Ublk,F,Bblk)
                    # accumulate: Y[:,j0:j0+bs-1] += 2*real(wk*Ublk[:,1:bs])
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Y[i,col]+=2.0*real(wk*Ublk[i,jj])
                        end
                    end
                    j0+=bs
                end
            elseif backend===:ls
                Ak=SparseMatrixCSC{ComplexF64,Int}(Az.m,Az.n,Az.colptr,Az.rowval,copy(Az.nzval))
                j0=1
                while j0<=m0
                    bs=min(blockrhs,m0-j0+1)
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Bblk[i,jj]=Q[i,col]
                        end
                    end
                    bs<blockrhs && (@views fill!(Bblk[:,bs+1:blockrhs],0.0+0.0im))
                    prob=LinearSolve.LinearProblem(Ak,Bblk;u0=Ublk)
                    lin=(ls_alg===nothing ? LinearSolve.init(prob) : LinearSolve.init(prob,ls_alg))
                    sol=LinearSolve.solve!(lin)
                    @inbounds for jj in 1:bs
                        col=j0+jj-1
                        for i in 1:n
                            Y[i,col]+=2.0*real(wk*sol.u[i,jj])
                        end
                    end
                    j0+=bs
                end
            else
                throw(ArgumentError("backend must be :lu or :ls"))
            end
            show_progress && next!(progress_k)
        end
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        mul!(R,A,Q)
        mul!(Aq,transpose(Q),R)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,A,X,λ,AX)
        fill!(inside,false)
        @inbounds for j in 1:m0
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate)
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
        ok=any(inside) && (maxres<tol)
        if two_hit
            if ok && (it==1 || prev_ok)
                debug && @printf("converged\n")
                break
            end
            prev_ok=ok
        else
            ok && break
        end
    end
    idx=Int[]
    @inbounds for j in 1:m0
        (λ[j]≥emin && λ[j]≤emax && res[j]<max(res_gate,10*tol)) && push!(idx,j)
    end
    isempty(idx) && return Float64[],Matrix{Float64}(undef,n,0),Float64[]
    λsel=λ[idx];p=sortperm(λsel);idx=idx[p];λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end















if abspath(PROGRAM_FILE)==@__FILE__

    BLAS.set_num_threads(Sys.CPU_THREADS) # to get max performance for testing

    # SOLVER PARAMETERS; NODES/POLES - CRITICAL TO GET SMALLEST POSSIBLE WITH LOWEST ITERATION COUNT
    nodes_general=32 # ellipse nodes
    do_general=true # for genereal real matrices, use full contour
    do_half=true # for symmetric real matrices, use half contour and real arithmetics
    do_distributed=true # distributed versions that assemble and solve each shifted system on the fly, for lower memory usage at the cost of more factorizations/solves; only implemented for general real and symmetric half-contour cases so far
    tol=1e-8
    show_progress=true # progress bars for all tests, for cleaner output set to false

    ns=[20_000] # matrix sizes to test (sparse)
    n=1000 # matrix size for dense test

    function lap1d(n::Int)
        d0=fill(2.0,n)
        d1=fill(-1.0,n-1)
        return spdiagm(-1=>d1,0=>d0,1=>d1)
    end

    lap1d_eigs_exact(n::Int)=[2-2*cos(k*pi/(n+1)) for k in 1:n]
    function max_abs_err(a::Vector{Float64},b::Vector{Float64})
        length(a)==length(b) || return Inf
        maximum(abs.(a.-b))
    end

    function test_feast_on_lap1d(;n=1000,emin=0.2,emax=0.8,m0=400,nodes=nodes_general,eta=0.8,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=general_feast_sparse_interval(A;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,debug=false,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        any(max_abs_err(λ,λ_ref)>1e-8) && @warn "max |Δλ| = $(max_abs_err(λ,λ_ref)) exceeds tolerance"
        #isempty(res) || @warn "max residual reported = $(maximum(res))"
        return λ,λ_ref,res
    end

    function test_sym_half_on_lap1d(;n=1000,emin=0.2,emax=0.8,m0=400,nodes=nodes_general,eta=0.8,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=sym_feast_sparse_interval_half(A;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "HALF: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        Δ=max_abs_err(λ,λ_ref)
        Δ>1e-8 && @warn "HALF: max |Δλ| = $Δ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_feast_on_dense_small(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,nodes=nodes_general,eta=0.8,tol=tol,maxiter=10,res_gate=1e-8,debug=false,backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A) # force it to be sparse to test the sparse FEAST implementation, even though it's a dense matrix; this is just for testing correctness and timing of the sparse version on small matrices.
        λ,X,res=general_feast_sparse_interval(As;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,maxiter=maxiter,tol=tol,res_gate=res_gate,debug=debug,is_hermitian=true,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        any(maxΔ>1e-8) && @warn "max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    # for 1d Laplacian do a sweep over matrix sizes and intervals, with timing. Try increasing matrix size for any anomalies. Careful to manually increase m0 and nodes as n grows to ensure good convergence.
    function bench_sweep(;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4] # interval sizes, can increase if needed to more or larger intervals.
        nodes=nodes_general
        eta=0.8
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            @info "Sweep: n=$n, emin=$emin, emax=$emax, expected eigs=$(round(Int,expected))"
            m0=ceil(Int,1.3*expected)
            test_feast_on_lap1d(n=n,emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        end
    end

    function bench_sweep_half(;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_general
        eta=0.8
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            @info "Half sweep: n=$n, emin=$emin, emax=$emax, expected eigs=$(round(Int,expected))"
            m0=ceil(Int,1.3*expected)
            test_sym_half_on_lap1d(n=n,emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,show_progress=show_progress)
        end
    end

    function test_feast_on_lap1d_distributed(;n=1000,emin=0.2,emax=0.8,m0=400,nodes=nodes_general,eta=0.8,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,blockrhs=0,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=general_feast_sparse_interval_distributed(A;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,debug=false,is_hermitian=true,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "DISTRIBUTED: expected $(length(λ_ref)) eigs but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "DISTRIBUTED: max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_lap1d_distributed(;n=1000,emin=0.2,emax=0.8,m0=400,nodes=nodes_general,eta=0.8,tol=tol,maxiter=30,res_gate=1e-6,backend=:lu,ls_alg=nothing,blockrhs=0,debug=false,show_progress::Bool=false)
        A=lap1d(n)
        λ_exact=lap1d_eigs_exact(n)
        idx=findall(x->(x>=emin && x<=emax),λ_exact)
        λ_ref=sort(λ_exact[idx])
        λ,X,res=sym_feast_sparse_interval_half_distributed(A;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "HALF/DIST: expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        Δ=max_abs_err(λ,λ_ref)
        Δ>1e-8 && @warn "HALF/DIST: max |Δλ| = $Δ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_dense_small(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,nodes=nodes_general,eta=0.8,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,debug=false,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res=sym_feast_sparse_interval_half(As;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "HALF(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "HALF(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_feast_on_dense_small_distributed(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,nodes=nodes_general,eta=0.8,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,blockrhs=0,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n;A[i,i]+=diagshift;end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res=general_feast_sparse_interval_distributed(As;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,debug=false,is_hermitian=true,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "DISTRIBUTED: expected $(length(λ_ref)) eigs but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "DISTRIBUTED: max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function test_sym_half_on_dense_small_distributed(;n=300,seed=1,density=1.0,diagshift=21.0,emin=20.5,emax=21.5,m0=40,nodes=nodes_general,eta=0.8,tol=tol,maxiter=10,res_gate=1e-8,backend=:lu,ls_alg=nothing,blockrhs=0,debug=false,show_progress::Bool=false)
        Random.seed!(seed)
        G=randn(n,n)
        if density<1.0
            M=rand(n,n).<=density
            G.=G.*M
        end
        A=0.5*(G+G')
        @inbounds for i in 1:n
            A[i,i]+=diagshift
        end
        E=eigen(Symmetric(A))
        λ_all=E.values
        idx=findall(x->(x>=emin && x<=emax),λ_all)
        λ_ref=sort(λ_all[idx])
        As=sparse(A)
        λ,X,res = sym_feast_sparse_interval_half_distributed(As;emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,debug=debug,show_progress=show_progress)
        λ=sort(λ)
        length(λ_ref)==length(λ) || @warn "HALF/DIST(DENSE): expected $(length(λ_ref)) eigenvalues but got $(length(λ))"
        maxΔ=(length(λ_ref)==length(λ)) ? maximum(abs.(λ.-λ_ref)) : Inf
        maxΔ>1e-8 && @warn "HALF/DIST(DENSE): max |Δλ| = $maxΔ exceeds tolerance"
        return λ,λ_ref,res
    end

    function bench_sweep_distributed(;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,blockrhs=0,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_general;eta=0.8;maxiter=10;res_gate=1e-6;emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0;emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            @info "Distributed sweep: n=$n, emin=$emin, emax=$emax, expected eigs=$(round(Int,expected))"
            m0=ceil(Int,1.3*expected)
            test_feast_on_lap1d_distributed(n=n,emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        end
    end

    function bench_sweep_half_distributed(;ns=[5_000,10_000,20_000,40_000],backend=:lu,ls_alg=nothing,blockrhs=32,show_progress::Bool=false)
        Δs=[0.1,0.2,0.4]
        nodes=nodes_general
        eta=0.8
        maxiter=10
        res_gate=1e-6
        emin0=0.2
        for n in ns,Δ in Δs
            emin=emin0
            emax=emin0+Δ
            expected=(n+1)/pi*(sqrt(emax)-sqrt(emin))
            @info "Distributed half sweep: n=$n, emin=$emin, emax=$emax, expected eigs=$(round(Int,expected))"
            m0=ceil(Int,1.3*expected)
            test_sym_half_on_lap1d_distributed(n=n,emin=emin,emax=emax,m0=m0,nodes=nodes,eta=eta,tol=tol,maxiter=maxiter,res_gate=res_gate,backend=backend,ls_alg=ls_alg,blockrhs=blockrhs,show_progress=show_progress)
        end
    end

    if do_half

    println()
    println("--------------------------------------------------")
    println("HALF-CONTOUR (sym_feast_sparse_interval_half)")
    println("--------------------------------------------------")

    println("SPARSE / HALF")
    s2=time()
    println("LinearSolve KLU")
    bench_sweep_half(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress)
    e=time()
    println("--------------------------------------------------")
    println("LinearSolve KLU backend:    ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------")

    println()
    
    s=time()
    println("DENSE / HALF")
    println("LU (default)")
    test_sym_half_on_dense_small(n=n,backend=:lu,show_progress=show_progress)
    e=time()
    println("--------------------------------------------------")
    println("LU backend:                 ",@sprintf("%.3f seconds",e-s))
    println("--------------------------------------------------")

        if do_distributed

        println()
        println("--------------------------------------------------")
        println("HALF-CONTOUR DISTRIBUTED (sym_feast_sparse_interval_half_distributed)")
        println("--------------------------------------------------")

        println("SPARSE / HALF / DISTRIBUTED")
        s=time()
        println("LinearSolve KLU (blockrhs=32)")
        bench_sweep_half_distributed(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),blockrhs=32,show_progress=show_progress)
        e=time()

        println("--------------------------------------------------")
        println("LS KLU blockrhs=32:         ",@sprintf("%.3f seconds",e-s))
        println("--------------------------------------------------")

        println()
        println("DENSE / HALF / DISTRIBUTED")
        
        s=time()
        test_sym_half_on_dense_small_distributed(n=n,backend=:lu,blockrhs=0,show_progress=show_progress)
        s1=time()
        test_sym_half_on_dense_small_distributed(n=n,backend=:lu,blockrhs=16,show_progress=show_progress)
        e=time()

        println("--------------------------------------------------")
        println("LU blockrhs=0:              ",@sprintf("%.3f seconds",s1-s))
        println("LU blockrhs=16:             ",@sprintf("%.3f seconds",e-s1))
        println("--------------------------------------------------")

        end

    end

    if do_general

    println()
    println("--------------------------------------------------")
    println("DEFAULT (general_feast_sparse_interval)")
    println("--------------------------------------------------")
    
    s=time()
    println("SPARSE")
    s2=time()
    println("LinearSolve KLU")
    bench_sweep(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),show_progress=show_progress) # fast for small ns
    e=time()
    # Analysis:
    println("--------------------------------------------------")
    println("LinearSolve KLU backend: ",@sprintf("%.3f seconds",e-s2))
    println("--------------------------------------------------") 
    println()
    
    s=time()
    println("DENSE")
    println("LU (default)")
    test_feast_on_dense_small(n=n,backend=:lu,show_progress=show_progress) # fast
    e=time()
    # Analysis:
    println("LU backend: ", @sprintf("%.3f seconds", e-s))
    println("--------------------------------------------------")

        if do_distributed

        println()
        println("--------------------------------------------------")
        println("DISTRIBUTED VERSION TESTS (general_feast_sparse_interval_distributed)")
        println("--------------------------------------------------")
        
        println("SPARSE / DISTRIBUTED")
        s3=time()
        println("LinearSolve KLU (blockrhs=32)")
        bench_sweep_distributed(ns=ns,backend=:ls,ls_alg=LinearSolve.KLUFactorization(),blockrhs=32,show_progress=show_progress)
        e=time()
        println("--------------------------------------------------")
        println("LS KLU blockrhs=32:     ",@sprintf("%.3f seconds",e-s3))
        println("--------------------------------------------------")

        println()
        println("DENSE / DISTRIBUTED")
        
        s=time()
        test_feast_on_dense_small_distributed(n=n,backend=:lu,blockrhs=0,show_progress=show_progress)
        s1=time()
        test_feast_on_dense_small_distributed(n=n,backend=:lu,blockrhs=16,show_progress=show_progress)
        e=time()

        println("--------------------------------------------------")
        println("LU blockrhs=0:      ",@sprintf("%.3f seconds",s1-s))
        println("LU blockrhs=16:     ",@sprintf("%.3f seconds",e-s1))
        println("--------------------------------------------------")

        end

    end

    end
