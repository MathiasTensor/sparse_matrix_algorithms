# ==============================================================================
# Chebyshev Filtered Subspace Iteration
# ==============================================================================
#
# This file implements polynomial-filtered subspace iteration for computing
# interior eigenpairs of large real symmetric sparse or matrix-free operators.
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
# Instead of using shifted linear solves (as in FEAST / rational filtering),
# this method constructs a polynomial filter p(A) that amplifies spectral
# components inside the desired interval while damping components outside it.
#
# The filtered subspace iteration step is
#
#     Y = p(A) Q
#
# followed by orthogonalization and Rayleigh–Ritz extraction.
#
# ------------------------------------------------------------------------------
# Mathematical Idea
# ------------------------------------------------------------------------------
#
# The matrix A is first rescaled to
#
#     Â = (A - cI) / d
#
# where
#
#     c = (smin + smax)/2,
#     d = (smax - smin)/2,
#
# so that the spectrum of Â lies approximately inside [-1, 1].
#
# A polynomial filter is then built in the Chebyshev basis:
#
#     p(x) = Σ_{k=0}^m c_k T_k(x),
#
# where T_k(x) is the Chebyshev polynomial of degree k.
#
# Three filter types are supported:
#
#   1. step   : approximates the indicator of [emin, emax]
#   2. delta  : approximates a localized peak near σ
#   3. smooth : approximates a smoothed interval window
#
# Jackson damping may be applied to reduce Gibbs oscillations and improve
# numerical stability.
#
# ------------------------------------------------------------------------------
# Implemented Variants
# ------------------------------------------------------------------------------
#
# 1. Sparse matrix version
#    - chebyshev_poly_real_symm_sparse(A, emin, emax; ...)
#    - uses sparse matrix–block products
#
# 2. Matrix-free version
#    - chebyshev_poly_real_symm_sparse(matvec!, n, emin, emax; ...)
#    - uses user-supplied matrix-vector or block-matrix-vector products
#
# ------------------------------------------------------------------------------
# Algorithm (per iteration)
# ------------------------------------------------------------------------------
#
# Given a search subspace Q ∈ ℝ^{n×m0}:
#
#   1. Apply polynomial filter:
#
#          Y = p(Â) Q
#
#      using the three-term Chebyshev recurrence
#
#          T₀(Â)Q = Q
#          T₁(Â)Q = ÂQ
#          T_{k+1}(Â)Q = 2Â T_k(Â)Q - T_{k-1}(Â)Q
#
#   2. Orthonormalize:
#
#          Q ← orth(Y)
#
#   3. Rayleigh–Ritz projection:
#
#          Aq = Qᵀ A Q
#
#   4. Dense projected eigensolve:
#
#          Aq u = λ u
#
#   5. Lift Ritz vectors:
#
#          X = Q u
#
#   6. Residual filtering / convergence check
#
# ------------------------------------------------------------------------------
# Key Parameters
# ------------------------------------------------------------------------------
#
# - m0          : subspace size (must exceed number of wanted eigenvalues)
# - degree      : polynomial degree of the filter
# - window_type : :step, :delta, or :smooth
# - jackson     : whether to use Jackson damping
# - smin, smax  : spectral bounds for rescaling
# - σ           : center for delta-like filters
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

###########################################################################
####################### MATRIX - FREE FUNCTIONS ###########################
###########################################################################

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

function make_thin_orth_workspace(::Type{Float64},n::Int,m0::Int)
    Qtmp=Matrix{Float64}(undef,n,m0)
    τ=Vector{Float64}(undef,min(n,m0))
    return Qtmp,τ
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
    residuals!(res::Vector{Float64},A::MatrixFreeRealSymOp,X::Matrix{Float64},λ::Vector{Float64},AX::Matrix{Float64})

Compute the residual norms `res[j] = ||A*X[:,j] - λ[j]*X[:,j]||` for each column `j` of `X`, where `A` is a real symmetric operator represented by a `MatrixFreeRealSymOp`. The matrix-vector product `A*X` is computed using the user-provided `matvec!` function, and the results are stored in `AX`.
"""
function residuals!(res::Vector{Float64},A::MatrixFreeRealSymOp,X::Matrix{Float64},λ::Vector{Float64},AX::Matrix{Float64})
    apply_A_block!(AX,A,X)
    n,m=size(X)
    @inbounds for j in 1:m
        lj=λ[j]
        s=0.0
        for i in 1:n
            v=AX[i,j]-lj*X[i,j]
            s+=v*v
        end
        res[j]=sqrt(s)
    end
    return nothing
end

"""
    gershgorin_bounds(A) -> (smin,smax)

Cheap spectral bounds for a real symmetric sparse matrix using Gershgorin discs. Safe but often loose.
"""
function gershgorin_bounds(A::SparseMatrixCSC{Float64,Int})
    n=size(A,1);size(A,2)==n||throw(ArgumentError("A must be square"))
    d=zeros(Float64,n)
    r=zeros(Float64,n)
    @inbounds for j in 1:n
        for t in A.colptr[j]:(A.colptr[j+1]-1)
            i=A.rowval[t]
            v=A.nzval[t]
            if i==j
                d[i]=v
            else
                r[i]+=abs(v)
            end
        end
    end
    return minimum(d.-r),maximum(d.+r)
end

########################################################
############### CHEBYSHEV FILTER COEFFS ################
########################################################

"""
    jackson_kernel(K)

Return Jackson damping coefficients `g_0,...,g_K`. Jackson damping suppresses Gibbs oscillations in truncated Chebyshev series,
trading sharpness for smoother, more stable polynomial filters.
Ref: A CHEBYSHEV–JACKSON SERIES BASED BLOCK SS–RR ALGORITHM FOR COMPUTING PARTIAL EIGENPAIRS OF REAL SYMMETRIC MATRICES, ZHONGXIAO J. AND TIANHANG L. https://arxiv.org/pdf/2508.20456
"""
function jackson_kernel(K::Int)
    K<=0 && return [1.0]
    g=zeros(Float64,K+1)
    α=pi/(K+2)
    sα=sin(α)
    @inbounds for k in 0:K
        g[k+1]=sin((k+1)*α)/((K+2)*sα)+(1-(k+1)/(K+2))*cos(k*α)
    end
    return g
end

"""
    cheb_delta_coeffs(σ̂,K;jackson=true)

Return Chebyshev coefficients for a delta-like peak centered at `σ̂∈[-1,1]`.

Mathematical form
-----------------
Approximates a sharply peaked kernel in the Chebyshev basis

    p(x) ≈ 1/π + 2/π Σ_{k≥1} cos(k arccos(σ̂)) T_k(x),

optionally Jackson-damped.
"""
function cheb_delta_coeffs(σ̂::Float64,K::Int;jackson::Bool=true)
    c=zeros(Float64,K+1)
    θ=acos(clamp(σ̂,-1.0,1.0))
    c[1]=1/pi
    @inbounds for k in 1:K
        c[k+1]=2*cos(k*θ)/pi
    end
    if jackson
        g=jackson_kernel(K)
        @inbounds for k in 0:K
            c[k+1]*=g[k+1]
        end
    end
    return c
end

"""
    cheb_window_coeffs(a,b,K;jackson=true)

Return Chebyshev coefficients for the indicator of the interval `[a,b]⊂[-1,1]`.

Mathematical form
-----------------
For the characteristic function χ_[a,b], the Chebyshev coefficients are

    c_0 = (β-α)/π,
    c_k = 2(sin(kβ)-sin(kα))/(kπ),  k≥1,

with `α=acos(b)`, `β=acos(a)`, assuming `a≤b`.
"""
function cheb_window_coeffs(a::Float64,b::Float64,K::Int;jackson::Bool=true)
    aa=max(-1.0,min(1.0,a))
    bb=max(-1.0,min(1.0,b))
    aa>bb && ((aa,bb)=(bb,aa))
    α=acos(bb)
    β=acos(aa)
    c=zeros(Float64,K+1)
    c[1]=(β-α)/pi
    @inbounds for k in 1:K
        c[k+1]=2*(sin(k*β)-sin(k*α))/(k*pi)
    end
    if jackson
        g=jackson_kernel(K)
        @inbounds for k in 0:K
            c[k+1]*=g[k+1]
        end
    end
    return c
end

"""
    cheb_smooth_window_coeffs(a,b,K;jackson=true,nsamp=4096,δ=0.02)

Return Chebyshev coefficients of a smoothed window on `[a,b]`, using tanh ramps.

Mathematical form
-----------------
The window is

    f(x)=½(1+tanh((x-a)/δ)) * ½(1-tanh((x-b)/δ)),

and coefficients are computed by numerical quadrature in the θ-variable
(`x=cos θ`). This often produces a more stable filter than the discontinuous
indicator.
"""
function cheb_smooth_window_coeffs(a::Float64,b::Float64,K::Int;jackson::Bool=true,nsamp::Int=4096,δ::Float64=0.02)
    aa=max(-1.0,min(1.0,a))
    bb=max(-1.0,min(1.0,b))
    aa>bb && ((aa,bb)=(bb,aa))
    c=zeros(Float64,K+1)
    w=pi/(nsamp-1)
    @inbounds for k in 0:K
        s=0.0
        for i in 1:nsamp
            θ=(i-1)*w
            xi=cos(θ)
            fi=0.5*(1+tanh((xi-aa)/δ))*0.5*(1-tanh((xi-bb)/δ))
            wt=(i==1 || i==nsamp) ? 0.5 : 1.0
            s+=wt*fi*cos(k*θ)
        end
        c[k+1]=(k==0 ? 1/pi : 2/pi)*w*s
    end
    if jackson
        g=jackson_kernel(K)
        @inbounds for k in 0:K
            c[k+1]*=g[k+1]
        end
    end
    return c
end

"""
    scaled_mul!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},X::Matrix{Float64},c::Float64,d::Float64,AX::Matrix{Float64})

Compute Y = ((A - cI)/d) * X using AX as workspace for A*X.
"""
function scaled_mul!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},X::Matrix{Float64},c::Float64,d::Float64,AX::Matrix{Float64})
    mul!(AX,A,X)
    n,m=size(X)
    invd=1/d
    @inbounds for j in 1:m
        for i in 1:n
            Y[i,j]=(AX[i,j]-c*X[i,j])*invd
        end
    end
    return nothing
end

"""
    scaled_mul!(Y::Matrix{Float64},A::MatrixFreeRealSymOp,X::Matrix{Float64},c::Float64,d::Float64,AX::Matrix{Float64})

Compute Y = ((A - cI)/d) * X for a matrix-free real symmetric operator A, using AX as workspace for A*X.
"""
function scaled_mul!(Y::Matrix{Float64},A::MatrixFreeRealSymOp,X::Matrix{Float64},c::Float64,d::Float64,AX::Matrix{Float64})
    apply_A_block!(AX,A,X)
    n,m=size(X)
    invd=1/d
    @inbounds for j in 1:m, i in 1:n
        Y[i,j]=(AX[i,j]-c*X[i,j])*invd
    end
    return nothing
end

"""
    apply_cheb_filter!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64})

Apply the Chebyshev polynomial filter of `A` with coefficients `coeffs` to the block `Q`.

The matrix is first rescaled to
    Â = (A-cI)/d
where
    c = (smin+smax)/2, d = (smax-smin)/2

so that σ(Â) ⊂ [-1,1].

The output is
    Y = Σ_{k=0}^m coeffs[k+1] T_k(Â) Q.
"""
function apply_cheb_filter!(Y::Matrix{Float64},A::SparseMatrixCSC{Float64,Int},Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64};multithreading::Bool=false)
    m=length(coeffs)-1
    n,blk=size(Q) # n is matrix size, blk is block size (number of vectors in Q)
    c=0.5*(smin+smax) # center of spectral interval
    d=0.5*(smax-smin) # half-width of spectral interval
    d>0 || throw(ArgumentError("need smin < smax"))
    # U0 = T0(Â)Q = Q
    copyto!(U0,Q) # workspace for T_k(Â) * Q
    fill!(Y,0.0) # workspace for the output
    c0=coeffs[1] # contribution of T_0(Â) * Q = Q
    @inbounds for j in 1:blk, i in 1:n
        Y[i,j]=c0*U0[i,j] # Y <- c_0 * U_0 = c_0 * T_0(Â) * Q = c_0 * Q
    end
    m==0 && return nothing # if the poly has degree 0, then p_0(x) = c_0m is a constant and we are done
    # the first chebyshev polynomial is T_1(x) = x (so T_1(Â) = Â), so U1 = T1(Â) * Q = Â * Q
    scaled_mul!(U1,A,Q,c,d,AX) # U1 = T1(Â) * Q = Â * Q
    c1=coeffs[2] # contribution of U1 = T_1(Â) * Q = Â * Q
    LinearAlgebra.BLAS.axpy!(c1,vec(U1),vec(Y)) # now we add the first (linear) term (c_1) to the output Y as Y <- Y + c_1 * U_1 = c_0 * Q + c_1 * Â * Q
    m==1 && return nothing # if order of polynomial is 1, then we are done
    # now we use the three-term recurrence for Chebyshev polynomials to compute U_k = T_k(Â) * Q for k=2,...,m:
    # U_k = 2 * Â * U_{k-1} - U_{k-2}, from T_{k+1}(x) = 2x * T_k(x)-T_{k-1}(x).
    # at the start of each iteration U0 hold T_{k-1}(Â) * Q and U1 holds T_k(Â) * Q, so we can compute U_{k+1} = 2Â * U_k - U_{k-1} = 2Â * U1 - U0
    invd=1/d
    @inbounds for k in 1:(m-1) # sequential 3-term reccurence
        # U2 = 2ÂU1 - U0
        mul!(AX,A,U1) # AX = A * U1 (not Â)
        for j in 1:blk
            for i in 1:n
            U2[i,j]=2*((AX[i,j]-c*U1[i,j])*invd)-U0[i,j] # U2 = 2Â * U1 - U0 = 2((A - cI) / d) * U1 - U0 = 2(A * U1 - c * U1)/d - U0; because A in the previous line is not scaled Â, we need to do the scaling here
        
            end
        end
        ck1=coeffs[k+2] # contribution of U_{k+1} = T_{k+1}(Â) * Q (shifted by 1 index because coeffs[1] is for T_0 ...)
        LinearAlgebra.BLAS.axpy!(ck1,vec(U2),vec(Y)) # add the contribution of U_{k+1} = T_{k+1}(Â) * Q to the output Y
        U0,U1,U2=U1,U2,U0 # reasing workspaces for the next iteration: now U0 holds T_k(Â) * Q and U1 holds T_{k+1}(Â) * Q for the next iteration
    end
    return nothing
end

"""
    apply_cheb_filter!(Y::Matrix{Float64},A::MatrixFreeRealSymOp,Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64})

Apply the Chebyshev polynomial filter of `A` with coefficients `coeffs` to the block `Q`, where now `A` is a linear map.

"""
function apply_cheb_filter!(Y::Matrix{Float64},A::MatrixFreeRealSymOp,Q::Matrix{Float64},coeffs::Vector{Float64},smin::Float64,smax::Float64,U0::Matrix{Float64},U1::Matrix{Float64},U2::Matrix{Float64},AX::Matrix{Float64})
    m=length(coeffs)-1
    n,blk=size(Q) # blk is the number of vector columns
    c=0.5*(smin+smax) # center of the window
    d=0.5*(smax-smin) # half width of the window
    copyto!(U0,Q) #U0 is the workspace before accumulating to Y (both Q and U0/1/2 are workspaces - there are 3 for U due to three term recurrence relation for the chebyshev polys)
    fill!(Y,0.0) # initialize the output
    c0=coeffs[1] # first coeff (constant) cheb poly
    @inbounds for j in 1:blk, i in 1:n
        Y[i,j]=c0*U0[i,j] # add the first contribution to Y from the constant term cheb poly
    end
    m==0 && return nothing # if constant, return (not the case mostly if not wrong)
    scaled_mul!(U1,A,Q,c,d,AX) # U1 = T1(Â) * Q = Â * Q (this rescales the window to [-1,1] and computes the 1st order chebyshev poly)
    LinearAlgebra.BLAS.axpy!(coeffs[2],vec(U1),vec(Y)) # add the 2nd order cheb poly with its weight/coefficient into the result accumulator Y
    m==1 && return nothing # if only first order return
    invd=1/d
    @inbounds for k in 1:(m-1) # sequential 3-term reccurence
        apply_A_block!(AX,A,U1) # apply matrix A (here it is a linear operator) to U1 and store into temp AX
        for j in 1:blk
            for i in 1:n
            U2[i,j]=2*((AX[i,j]-c*U1[i,j])*invd)-U0[i,j] # U2 = 2Â * U1 - U0 = 2((A - cI) / d) * U1 - U0 = 2(A * U1 - c * U1)/d - U0; because A in the previous line is not scaled Â, we need to do the scaling here
            end
        end
        LinearAlgebra.BLAS.axpy!(coeffs[k+2],vec(U2),vec(Y)) # accumulate the new k-th term into the output
        U0,U1,U2=U1,U2,U0 # reshift the indexes: T0,T1,T2 -> T3, next iter T1,T2,T3 -> T4 etc.
    end
    return nothing
end

"""
    chebyshev_poly_real_symm_sparse(Ain::SparseMatrixCSC{Float64,Int},emin::Float64,emax::Float64;smin::Union{Float64,Nothing}=nothing,smax::Union{Float64,Nothing}=nothing,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step)

Compute eigenpairs of a real symmetric sparse matrix with eigenvalues in [emin,emax]
using Chebyshev polynomial filtered subspace iteration.

Inputs
- `Ain`: real symmetric sparse matrix (in CSC format)
- `emin`, `emax`: target interval for eigenvalues (must satisfy emin<emax)

Parameters
- `smin`, `smax`: spectral bounds for the matrix (must satisfy smin<smax and contain [emin,emax]); if not provided, will be estimated with Gershgorin bounds (cheap but often loose, which can degrade filter quality)
- `σ`:
    In case of the window_type == :delta, this is the center of the delta function.
- `m0`:
    Search subspace dimension. Must exceed the number of eigenvalues in the interval.
- `degree`:
    Degree of the Chebyshev window polynomial. Larger degree => sharper spectral filter.
- `jackson`:
    If true, use Jackson damping to reduce Gibbs oscillations.
- `two_hit`:
    If true, require 2 consecutive iterations with maxres<tol (except immediate break at it=1).
- `window_type::Symbol`
    If :delta, approximates a "wide" delta and if :step gives the step function in the given wanted energy interval. Also can use :smooth to give a smoothed indicator function on the interval for better convergence.

Returns
- `λsel`: selected eigenvalues in [emin,emax], sorted
- `Xsel`: corresponding eigenvectors
- `ressel`: residual norms
"""
function chebyshev_poly_real_symm_sparse(Ain::SparseMatrixCSC{Float64,Int},emin::Float64,emax::Float64;smin::Union{Float64,Nothing}=nothing,smax::Union{Float64,Nothing}=nothing,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step)

    issymmetric(Ain) || throw(ArgumentError("Chebyshev filtered solver requires a real symmetric matrix"))
    (emin<emax) || throw(ArgumentError("need emin<emax"))
    m0>0 || throw(ArgumentError("m0 must be positive"))
    degree>=0 || throw(ArgumentError("degree must be nonnegative"))
    n=size(Ain,1)
    size(Ain,2)==n || throw(ArgumentError("A must be square"))
    m0=min(m0,n)
    A=Ain
    if smin===nothing || smax===nothing
        smin_,smax_=gershgorin_bounds(A) # cheap spectral bounds
    else
        smin_=smin
        smax_=smax
    end
    (emin>=smin_ && emax<=smax_) || @warn "target interval not contained in supplied spectral bounds; filter quality may degrade"
    # target interval mapped to [-1,1]
    c=0.5*(smin_+smax_) # center of spectral interval
    d=0.5*(smax_-smin_) # half-width of spectral interval
    a=(emin-c)/d # left endpoint of target interval mapped to [-1,1]
    b=(emax-c)/d # right endpoint of target interval mapped to [-1,1]
    -1.0<=a<b<=1.0 || throw(ArgumentError("mapped interval [$a,$b] not inside [-1,1]; provide better spectral bounds"))
    σ̂=(σ-c)/d
     # give the window coefficients for the mapped interval
    if window_type==:step
        coeffs=cheb_window_coeffs(a,b,degree;jackson=jackson)
    elseif window_type==:delta
        coeffs=cheb_delta_coeffs(σ̂,degree;jackson=jackson)
    elseif window_type==:smooth
        coeffs=cheb_smooth_window_coeffs(a,b,degree;jackson=jackson)
    else
        @error("Unknown window type, possible are :step, :delta, and :smooth")
    end
    Random.seed!(seed)
    Q=randn(n,m0) # initialize a random block of m0 vectors; will be replaced by the filtered subspace
    Y=Matrix{Float64}(undef,n,m0) # workspace for the filtered block Y = p(Â)Q
    U0=Matrix{Float64}(undef,n,m0) # workspace for T_k(Â)Q in the Chebyshev recurrence
    U1=Matrix{Float64}(undef,n,m0) # workspace for T_k(Â)Q in the Chebyshev recurrence
    U2=Matrix{Float64}(undef,n,m0) # workspace for T_k(Â)Q in the Chebyshev recurrence
    AXf=Matrix{Float64}(undef,n,m0) # workspace for A*Q in the Chebyshev recurrence
    # RR workspaces
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0) # workspaces for thin orthogonalization of Q
    R=Matrix{Float64}(undef,n,m0) # workspace for A*Q in the Rayleigh-Ritz step
    Aq=Matrix{Float64}(undef,m0,m0) # workspace for the Rayleigh quotient matrix Q'*A*Q
    Xq=Matrix{Float64}(undef,m0,m0) # workspace for Ritz vectors in the Rayleigh-Ritz step
    λ=Vector{Float64}(undef,m0) # workspace for Ritz values in the Rayleigh-Ritz step
    X=Matrix{Float64}(undef,n,m0) # workspace for Ritz vectors in the original space
    AX=Matrix{Float64}(undef,n,m0) # workspace for A*X in the residual computation
    res=Vector{Float64}(undef,m0) # workspace for residual norms
    inside=falses(m0) # workspace for tracking which Ritz pairs are inside the target interval and have residuals below res_gate
    # ALGORITHM
    # For it in 1:maxiter
    #     1. Q = orthonormalize(Q)
    #     2. Y = p(Â)Q - apply the Chebyshev filter to the block Q; Y is the filtered block
    #     3. Q = Y - replace Q by the filtered block; will be orthogonalized in the next iteration
    #     4. Q = orthonormalize(Q) - orthogonalize the filtered block Q
    #     5. R = A*Q - compute A*Q for the Rayleigh-Ritz step
    #     6. Aq = Q'*A*Q - compute the Rayleigh quotient matrix
    #     7. E = eigen(Symmetric(Aq)) - Ritz values/vectors of Aq; E.values are sorted in ascending order
    #     8. X = Q*E.vectors - Ritz vectors in the original space
    #     9. res = residuals(A,X,λ,AX) - compute residual norms for the Ritz pairs
    #     10. track which Ritz pairs are inside the target interval and have residuals below res_gate
    #     11. maxres = maximum residual norm among the Ritz pairs inside the target interval
    #     12. if maxres < tol for 2 consecutive iterations (it two_hit, except immediate break at it=1), then break
    # end
    prev_ok=false # flag to allow termination after 2 consecutive iterations with maxres<tol (except immediate break at it=1)
    for it in 1:maxiter
        apply_cheb_filter!(Y,A,Q,coeffs,smin_,smax_,U0,U1,U2,AXf) # Y = p(Â)Q - filtered block
        Q.=Y # replace Q by the filtered block; will be orthogonalized in the next iteration
        thin_orth!(Q,Qtmp,Z) # orthogonalize the filtered block Q
        mul!(R,A,Q) # compute A*Q for the Rayleigh-Ritz step
        mul!(Aq,transpose(Q),R) # Aq = Q'*A*Q - Rayleigh quotient matrix
        E=eigen!(Symmetric(Aq)) # Ritz values/vectors of Aq; E.values are sorted in ascending order
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq) # Ritz vectors in the original space
        residuals!(res,A,X,λ,AX) # compute residual norms for the Ritz pairs
        
        fill!(inside,false)
        @inbounds for j in 1:m0 
            inside[j]=(λ[j]≥emin && λ[j]≤emax && res[j]<res_gate) # track which Ritz pairs are inside the target interval and have residuals below res_gate
        end
        maxres=any(inside) ? maximum(@view res[inside]) : Inf # maximum residual norm among the Ritz pairs inside the target interval
        debug && @printf("it=%d inside=%d maxres=%.3e\n",it,count(inside),maxres)
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
    chebyshev_poly_real_symm_sparse(matvec!::F,n::Int,emin::Float64,emax::Float64;matblock::Union{G,Nothing}=nothing,smin::Union{Float64,Nothing}=nothing,smax::Union{Float64,Nothing}=nothing,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step) where {F,G}

For details on parameters see the `chebyshev_poly_real_symm_sparse` dispatch with the concrete matrix `Ain`. The input is the callnack with signature `f(y,x) := y = A*x` and `n` is the matrix dimension of the linear operator.

Additional Inputs
- `matvec!::F`: a user-provided function that computes the matrix-vector product `y = A*x` for the linear operator `A` whose eigenpairs we want to compute; the function should have the signature `f(y,x) := y = A*x` where `y` and `x` are vectors of length `n`.

Additional Parameters
- `matblock::Union{G,Nothing}`: optional additional argument to be passed to the `matvec!` function, for example to allow for a block matrix-vector product; if not needed, can be left as `nothing` (default).

"""
function chebyshev_poly_real_symm_sparse(matvec!::F,n::Int,emin::Float64,emax::Float64;matblock::Union{G,Nothing}=nothing,smin::Union{Float64,Nothing}=nothing,smax::Union{Float64,Nothing}=nothing,σ::Float64=0.0,m0::Int=200,degree::Int=80,maxiter::Int=8,tol::Float64=1e-10,res_gate::Float64=1e-6,jackson::Bool=true,debug::Bool=false,two_hit::Bool=true,seed::Int=1,window_type::Symbol=:step) where {F,G}
    Ain=MatrixFreeRealSymOp(n,matvec!,matblock)
    n=size(Ain,1)
    m0=min(m0,n)
    if smin===nothing || smax===nothing
        smin_,smax_=gershgorin_bounds(Ain)
    else
        smin_=smin
        smax_=smax
    end
    c=0.5*(smin_+smax_)
    d=0.5*(smax_-smin_)
    a=(emin-c)/d
    b=(emax-c)/d
    σ̂=(σ-c)/d
    -1.0<=a<b<=1.0 || throw(ArgumentError("mapped interval [$a,$b] not inside [-1,1]"))
    if window_type==:step
        coeffs=cheb_window_coeffs(a,b,degree;jackson=jackson)
    elseif window_type==:delta
        coeffs=cheb_delta_coeffs(σ̂,degree;jackson=jackson)
    elseif window_type==:smooth
        coeffs=cheb_smooth_window_coeffs(a,b,degree;jackson=jackson)
    else
        @error("Unknown window type, possible are :step, :delta, and :smooth")
    end
    Random.seed!(seed)
    Q=randn(n,m0)
    Y=Matrix{Float64}(undef,n,m0)
    U0=Matrix{Float64}(undef,n,m0)
    U1=Matrix{Float64}(undef,n,m0)
    U2=Matrix{Float64}(undef,n,m0)
    AXf=Matrix{Float64}(undef,n,m0)
    Qtmp,Z=make_thin_orth_workspace(Float64,n,m0)
    R=Matrix{Float64}(undef,n,m0)
    Aq=Matrix{Float64}(undef,m0,m0)
    Xq=Matrix{Float64}(undef,m0,m0)
    λ=Vector{Float64}(undef,m0)
    X=Matrix{Float64}(undef,n,m0)
    AX=Matrix{Float64}(undef,n,m0)
    res=Vector{Float64}(undef,m0)
    inside=falses(m0)
    prev_ok=false
    for it in 1:maxiter
        apply_cheb_filter!(Y,Ain,Q,coeffs,smin_,smax_,U0,U1,U2,AXf)
        Q.=Y
        thin_orth!(Q,Qtmp,Z)
        apply_A_block!(R,Ain,Q)
        mul!(Aq,transpose(Q),R)
        E=eigen!(Symmetric(Aq))
        λ.=E.values
        Xq.=E.vectors
        mul!(X,Q,Xq)
        residuals!(res,Ain,X,λ,AX)
        fill!(inside, false)
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
    λsel=λ[idx]
    p=sortperm(λsel)
    idx=idx[p]
    λsel=λsel[p]
    return λsel,X[:,idx],res[idx]
end







#######################################
############### TESTING ###############
#######################################

if abspath(PROGRAM_FILE)==@__FILE__

    n=10_000 # size of the matrix
    BLAS.set_num_threads(Sys.CPU_THREADS) # to get max performance for testing
    degrees_chebyshev=[16,32,48,64,96,128] # degrees of the chebyshev polynomial approximating the delta function filter
    counts_chebyshev=reverse([100,150,200,250,500,1000]) # how many eigenvalues we want in the delta window, more eigenvealues (m0), less degree we need
    do_matrix_free=true
    do_matrix_concrete=false

    # 1D Laplacian tridiag(-1,2,-1)
    function lap1d(n::Int)
        d0=fill(2.0,n)
        d1=fill(-1.0,n-1)
        return spdiagm(-1=>d1,0=>d0,1=>d1)
    end
    function lap1d_matvec!(y,x)
        n=length(x)
        @inbounds begin
            y[1]=2*x[1]-x[2]
            for i in 2:n-1
                y[i]=2*x[i]-x[i-1]-x[i+1]
            end
            y[n]=2*x[n]-x[n-1]
        end
    end
    function lap1d_block_matvec!(Y,X)
        n,m=size(X)
        @inbounds begin
            for j in 1:m
                Y[1,j]=2X[1,j]-X[2,j]
            end
            for i in 2:n-1
                for j in 1:m
                    Y[i,j]=2*X[i,j]-X[i-1,j]-X[i+1,j]
                end
            end
            for j in 1:m
                Y[n,j]=2*X[n,j]-X[n-1,j]
            end
        end
        return Y
    end
    lap1d_eigs_exact(n::Int)=[2-2*cos(k*pi/(n+1)) for k in 1:n]

    function nearest_k_to_sigma(vals::Vector{Float64},σ::Float64,k::Int)
        p=sortperm(abs.(vals .- σ))
        out=vals[p[1:min(k,length(vals))]]
        sort!(out)
        return out
    end
    function local_match_error(λ::Vector{Float64},λexact::Vector{Float64},σ::Float64)
        k=min(length(λ),length(λexact))
        k==0 && return Inf
        a=nearest_k_to_sigma(λ,σ,k)
        b=nearest_k_to_sigma(λexact,σ,k)
        return maximum(abs.(a.-b))
    end
    function match_stats_sorted(ref::Vector{Float64},got::Vector{Float64};tol::Float64=1e-8)
        nref=length(ref)
        ngot=length(got)
        ref_hit=falses(nref)
        got_hit=falses(ngot)
        i=1
        j=1
        while i<=nref && j<=ngot
            d=got[j]-ref[i]
            if abs(d)<=tol
                ref_hit[i]=true
                got_hit[j]=true
                i+=1
                j+=1
            elseif d<0
                j+=1
            else
                i+=1
            end
        end
        missed_ref=findall(!,ref_hit)
        extra_got=findall(!,got_hit)
        return (nref=nref,ngot=ngot,nmatched_ref=count(ref_hit),nmatched_got=count(got_hit),nmissed=length(missed_ref),nextra=length(extra_got),missed_idx=missed_ref,extra_idx=extra_got,ref_hit=ref_hit,got_hit=got_hit)
    end
    function classify_against_full_spectrum(λgot::Vector{Float64},λall::Vector{Float64};tol::Float64=1e-8)
        ng=length(λgot)
        na=length(λall)
        got_legit=falses(ng)
        i=1
        j=1
        while i<=ng && j<=na
            d=λgot[i]-λall[j]
            if abs(d)<=tol
                got_legit[i]=true
                i+=1
                j+=1
            elseif d<0
                i+=1
            else
                j+=1
            end
        end
        legit_idx=findall(got_legit)
        spurious_idx=findall(!,got_legit)
        return (nlegit=length(legit_idx),nspurious=length(spurious_idx),legit_idx=legit_idx,spurious_idx=spurious_idx,got_legit=got_legit)
    end
    function classify_local_extras(λref::Vector{Float64},λgot::Vector{Float64},λall::Vector{Float64};tol::Float64=1e-8)
        local_stats=match_stats_sorted(λref,λgot;tol=tol)
        full_stats=classify_against_full_spectrum(λgot,λall;tol=tol)
        legit_extra_idx=Int[]
        spurious_extra_idx=Int[]
        local_extra_mask=falses(length(λgot))
        local_extra_mask[local_stats.extra_idx].=true
        for j in local_stats.extra_idx
            if full_stats.got_legit[j]
                push!(legit_extra_idx,j)
            else
                push!(spurious_extra_idx,j)
            end
        end
        return (nmatched_target=local_stats.nmatched_ref,nmissed_target=local_stats.nmissed,nextra=local_stats.nextra,nlegit_extra=length(legit_extra_idx),nspurious_extra=length(spurious_extra_idx),missed_target_idx=local_stats.missed_idx,extra_idx=local_stats.extra_idx,legit_extra_idx=legit_extra_idx,spurious_extra_idx=spurious_extra_idx)
    end
    function lap1d_interval_by_count(n::Int,nev_target::Int;center_frac::Float64=0.5,pad::Int=2)
        λ=lap1d_eigs_exact(n)
        ic=clamp(round(Int,center_frac*n),1,n)
        h=nev_target÷2
        i1=max(1,ic-h)
        i2=min(n,i1+nev_target-1)
        i1=max(1,i2-nev_target+1)
        emin=λ[max(1,i1-pad)]
        emax=λ[min(n,i2+pad)]
        σ=0.5*(λ[i1]+λ[i2])
        λref=λ[i1:i2]
        return emin,emax,σ,collect(λref),(i1,i2)
    end
    function test_cheb_delta_on_lap1d_count(matvec!,matblock!;n=5000,nev_target=50,center_frac=0.5,m0_pad=1.5,degree=64,tol=1e-8,maxiter=30,res_gate=1e-6,show_progress=false,window_type=:step,matrix_free=false)
        smin=0.0
        smax=4.0
        A=lap1d(n)
        λall=lap1d_eigs_exact(n)
        emin,emax,σ,λref,(i1,i2)=lap1d_interval_by_count(n,nev_target;center_frac=center_frac)
        m0=min(n,ceil(Int,m0_pad*nev_target))

        println("CHEBΔ count-test: n=",n," nev_target=",nev_target,
                " idx=[",i1,",",i2,"] interval=[",emin,", ",emax,"] σ=",σ,
                " m0=",m0," degree=",degree)
        flush(stdout)
        if matrix_free
            λ,X,res=chebyshev_poly_real_symm_sparse(lap1d_matvec!,n,emin,emax;matblock=lap1d_block_matvec!,σ=σ,smin=smin,smax=smax,m0=m0,degree=degree,maxiter=maxiter,tol=tol,res_gate=res_gate,jackson=true,window_type=window_type)
        else
            λ,X,res=chebyshev_poly_real_symm_sparse(A,emin,emax;σ=σ,smin=smin,smax=smax,m0=m0,degree=degree,maxiter=maxiter,tol=tol,res_gate=res_gate,jackson=true, window_type=window_type)
        end
        λ=sort(λ)
        λref=sort(λref)
        λall=sort(λall)
        cls=classify_local_extras(λref,λ,λall;tol=1e-8)
        println("matched target = ",cls.nmatched_target," / ",length(λref))
        println("missed target  = ",cls.nmissed_target)
        println("extra total    = ",cls.nextra)
        println("legit extra    = ",cls.nlegit_extra)
        println("spurious extra = ",cls.nspurious_extra)
        Δloc=local_match_error(λ,λref,σ)
        rmax=isempty(res) ? Inf : maximum(res)
        meta=(emin=emin,emax=emax,σ=σ,m0=m0,i1=i1,i2=i2,Δloc=Δloc,rmax=rmax,nmatched_target=cls.nmatched_target,nmissed_target=cls.nmissed_target,nextra=cls.nextra,nlegit_extra=cls.nlegit_extra,nspurious_extra=cls.nspurious_extra)
        return λ,λref,res,meta
    end

    function bench_sweep_cheb_counts(;n=5000,counts=[20,50,100,150],degrees=[16,32,48,64,96,128],center_frac=0.5,maxiter=30,tol=1e-8,res_gate=1e-6,m0_pad=1.5,window_type=:step,matrix_free::Bool=false)

        println("==========================================================================")
        println("CHEB DELTA local-count benchmark")
        println("==========================================================================")
        @printf("%8s %8s %8s %8s %8s %8s %8s %12s\n",
                "nev","deg","m0","got","match","lextra","spur","localerr")
        println("--------------------------------------------------------------------------")

        for nev in counts
            for deg in degrees
                t=@elapsed begin
                    λ,λref,res,meta=test_cheb_delta_on_lap1d_count(lap1d_matvec!,lap1d_block_matvec!,
                    n=n,nev_target=nev,center_frac=center_frac,
                    m0_pad=m0_pad,degree=deg,
                    tol=tol,maxiter=maxiter,res_gate=res_gate,
                    show_progress=false,window_type=window_type,matrix_free=matrix_free)
                    global _λ=λ
                    global _λref=λref
                    global _res=res
                    global _meta=meta
                end
                @printf("%8d %8d %8d %8d %8d %8d %8d %12.3e\n",
                        nev,deg,_meta.m0,length(_λ),
                        _meta.nmatched_target,
                        _meta.nlegit_extra,
                        _meta.nspurious_extra,
                        _meta.Δloc)
                println("time[s] = ", @sprintf("%.4f", t))
                flush(stdout)
                nev==_meta.nmatched_target && break # if we got all the target eigenvalues, no need to keep increasing degree for this count
            end
            println("--------------------------------------------------------------------------")
        end
    end

    if do_matrix_free
    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV DELTA FILTER MATRIX FREE")
    println("--------------------------------------------------")

    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:delta,matrix_free=true)

    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV STEP WINDOW MATRIX FREE")
    println("--------------------------------------------------")

    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:step,matrix_free=true)

    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV SMOOTH WINDOW MATRIX FREE")
    println("--------------------------------------------------")

    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:smooth,matrix_free=true)

    end

    if do_matrix_concrete
    
    println()
    println("--------------------------------------------------")
    println("CHEBYSHEV DELTA FILTER SPARSE")
    println("--------------------------------------------------")

    println("SPARSE / CHEB DELTA")
    
    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:delta,matrix_free=false)

    println("--------------------------------------------------")
    println("CHEBYSHEV STEP FILTER SPARSE")
    println("--------------------------------------------------")

    println("SPARSE / CHEB STEP")
    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:step,matrix_free=false)

    println("--------------------------------------------------")
    println("CHEBYSHEV SMOOTH FILTER SPARSE")
    println("--------------------------------------------------")

    println("SPARSE / CHEB SMOOTH")
    bench_sweep_cheb_counts(n=n,counts=counts_chebyshev,degrees=degrees_chebyshev,center_frac=0.5,maxiter=50,tol=1e-8,res_gate=1e-6,m0_pad=1.3,window_type=:smooth,matrix_free=false)

    end

end

#########################
#### RESULTS SUMMARY ####
#########################

#=
--------------------------------------------------
CHEBYSHEV DELTA FILTER MATRIX FREE
--------------------------------------------------
==========================================================================
CHEB DELTA local-count benchmark
==========================================================================
     nev      deg       m0      got    match   lextra     spur     localerr
--------------------------------------------------------------------------
CHEBΔ count-test: n=10000 nev_target=1000 idx=[4500,5499] interval=[1.685610886680983, 2.3131481613293645] σ=1.9993794622045669 m0=1300 degree=32
matched target = 1000 / 1000
missed target  = 0
extra total    = 3
legit extra    = 3
spurious extra = 0
    1000       32     1300     1003     1000        3        0    4.441e-15
time[s] = 40.5311
--------------------------------------------------------------------------
=#
