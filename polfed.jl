##############################################
################### POLFED ###################
##############################################

#=
background
----------------------
Let A be real symmetric with spectrum in [smin,smax]. Define the affine map

    Â = (A-cI)/d,   c=(smin+smax)/2,   d=(smax-smin)/2,

so that spec(Â) ⊂ [-1,1].
[Implemented through the rescaling inside `_build_coeffs(...)` and
`apply_poly_block!(...)`.]

Choose a polynomial filter p(x) that is large on the mapped target interval
[a,b] = ((emin-c)/d,(emax-c)/d) and small outside it. In this code p is
expanded in Chebyshev polynomials:

    p(x) = Σ_{k=0}^K α_k T_k(x).

[Filter coefficients are built by `_build_coeffs(...)`, with concrete choices
`cheb_window_coeffs(...)`, `cheb_smooth_window_coeffs(...)`, or
`cheb_delta_coeffs(...)`, optionally damped by `jackson_kernel(...)`.]

Applying p(Â) to a block Q amplifies components belonging to eigenvectors
whose eigenvalues lie in the target window.
[Implemented by `apply_poly_block!(...)` via a block Clenshaw recurrence.]

We then build a block Lanczos-type basis in the filtered operator:

    Q_1 given,
    W_j = p(Â)Q_j - Q_{j-1}B_{j-1}^T,
    A_j = Q_j^T W_j,
    W_j ← W_j - Q_j A_j,
    orthogonalize W_j,
    W_j = Q_{j+1} B_j.

[The initial block Q₁ is generated in `polfed(...)` by
`orth_block_polfed!(randn(...))`.
The full recurrence is implemented in `build_full_basis!(...)`.
Orthogonalization/rank truncation is handled by `reorth_full!(...)`,
`orth_against!(...)`, and `block_qr_factor(...)`.]

This yields a reduced symmetric block tridiagonal matrix T satisfying

    p(Â) V_m ≈ V_m T,

where V_m=[Q_1,...,Q_m] is the accumulated orthonormal basis.
[The block basis is stored in `Qhist`, and the reduced matrix data are stored in
`Ablocks` and `Bblocks` inside `build_full_basis!(...)`.
The explicit dense reduced matrix T is assembled by `assemble_T(...)`.]

Then solve the dense reduced problem

    T u = θ u,

[Implemented in `extract_pairs(...)` via `E = eigen(T)`.]

reconstruct Ritz vectors of the original problem

    x = V_m u,

[Implemented in `extract_pairs(...)` by `build_X!(...)`.]

and evaluate original Rayleigh quotients and residuals with A:

    λ = x^T A x,
    r = ||Ax - λx||_2.

[Implemented by `rayleigh!(...)`, called inside `extract_pairs(...)`.]

Finally keep only vectors with λ ∈ [emin,emax], optionally refine them by a
Rayleigh–Ritz step in the original operator, and return the best pairs.
[Interval filtering, residual sorting, and final selection are done in
`extract_pairs(...)`.
Optional original-space refinement is performed by `refine_original_rr(...)`.
The top-level is `polfed(...)`.]

Parameters
----------------------------------------
nev
    Number of desired eigenpairs in [emin,emax].

s
    Block size. Each recurrence step contributes about s basis vectors.

mmax
    Number of block steps. The total reduced-space dimension is roughly

        M ≈ s*mmax

    up to rank loss from QR truncation.

degree
    Degree K of the Chebyshev filter polynomial. Larger degree sharpens the
    interval filter but increases the cost of each apply_poly_block! call.

overextract
    Number of reduced Ritz vectors reconstructed relative to nev. The code
    reconstructs roughly max(overextract*nev,nev+extra_keep) candidates before
    interval filtering and residual-based pruning.

Window choices
--------------
:step
    Indicator-like polynomial on [a,b].

:delta
    Peak-like polynomial around sigma.

:smooth
    Smoothed indicator on [a,b] using tanh ramps; often more stable than a hard
    step window.

REF: Polynomially Filtered Exact Diagonalization Approach to Many-Body Localization; Sierant P., Lewenstein M., Zakrzewski J. https://arxiv.org/abs/2005.09534

# BROUGHT TO YOU BY THE POWER OF CHATGPT-5.2
# ==============================================================================
=#

using LinearAlgebra,SparseArrays,Random,ProgressMeter

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

"""
    apply_A_vec!(y,A,x)

Compute `y ← A*x` for a sparse explicit matrix A.

This is the primitive operator application used by the polynomial recurrence.
"""
function apply_A_vec!(y::AbstractVector{Float64},A::SparseMatrixCSC{Float64,Int},x::AbstractVector{Float64})
    mul!(y,A,x)
    return y
end

"""
    apply_A_vec!(y,Aop,x)

Compute `y ← A*x` for a matrix-free operator `Aop`.
"""
function apply_A_vec!(y::AbstractVector{Float64},A::MatrixFreeRealSymOp,x::AbstractVector{Float64})
    A.matvec!(y,x)
    return y
end

"""
    apply_A_block!(Y,A,X)

Compute `Y ← A*X` for a sparse explicit matrix A.

If X has columns `x_j`, then the output columns are `Ax_j`.
"""
function apply_A_block!(Y::AbstractMatrix{Float64},A::SparseMatrixCSC{Float64,Int},X::AbstractMatrix{Float64})
    mul!(Y,A,X)
    return Y
end

"""
    apply_A_block!(Y,Aop,X)

Compute `Y ← A*X` for a matrix-free operator.

If `Aop.matblock!` exists it is used directly; otherwise the action is applied
columnwise in parallel.
"""
function apply_A_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})
    if !isnothing(A.matblock!)
        A.matblock!(Y,X)
    else
        b=size(X,2)
        Threads.@threads for j in 1:b
            A.matvec!(view(Y,:,j),view(X,:,j))
        end
    end
    return Y
end

"""
    scaled_apply_vec!(y,A,x,c,d,ax)

Compute

    y = (A*x - c*x)/d

using temporary storage `ax` for `A*x`. This applies the affine spectral normalization

    Â = (A-cI)/d,

which maps the spectral interval [smin,smax] to [-1,1], the natural domain of
Chebyshev polynomials.
"""
function scaled_apply_vec!(y::AbstractVector{Float64},A,x,c::Float64,d::Float64,ax::AbstractVector{Float64})
    apply_A_vec!(ax,A,x)
    invd=1/d
    @inbounds @simd for i in eachindex(x)
        y[i]=(ax[i]-c*x[i])*invd
    end
    return y
end

"""
    gershgorin_bounds(A)

Return Gershgorin lower/upper bounds for the spectrum of a real sparse matrix A. Every eigenvalue λ of A lies in at least one Gershgorin disk. For a real
symmetric matrix this yields a crude enclosing interval

    [min_i(a_ii-r_i), max_i(a_ii+r_i)],

where `r_i = Σ_{j≠i}|a_ij|`. This is used only to estimate spectral bounds for the Chebyshev rescaling.
"""
function gershgorin_bounds(A::SparseMatrixCSC{Float64,Int})
    n=size(A,1)
    d=diag(A)
    rowsums=zeros(Float64,n)
    @inbounds for j in 1:n
        for p in nzrange(A,j)
            i=A.rowval[p]
            v=abs(A.nzval[p])
            i==j || (rowsums[i]+=v)
        end
    end
    return minimum(d.-rowsums),maximum(d.+rowsums)
end

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
    _build_coeffs(degree,smin,smax,sigma,emin,emax;window_type=:step,jackson=true)

Build Chebyshev coefficients for the chosen filter.

Mapping
-------
The physical spectral interval is mapped by

    x = (λ-c)/d,  c=(smin+smax)/2, d=(smax-smin)/2,

so `sigma`, `emin`, and `emax` are first converted to the normalized Chebyshev
domain `[-1,1]`.
"""
function _build_coeffs(degree,smin,smax,sigma,emin,emax;window_type::Symbol=:step,jackson::Bool=true)
    c=0.5*(smin+smax)
    d=0.5*(smax-smin)
    σ̂=(sigma-c)/d
    a=(emin-c)/d
    b=(emax-c)/d
    return window_type===:step   ? cheb_window_coeffs(a,b,degree;jackson=jackson) :
           window_type===:delta  ? cheb_delta_coeffs(σ̂,degree;jackson=jackson) :
           window_type===:smooth ? cheb_smooth_window_coeffs(a,b,degree;jackson=jackson) :
           error("Unknown window type: $window_type")
end

"""
    POLFEDWorkspace(n,b)

Workspace for block Clenshaw evaluation of the Chebyshev filter.

Fields
------
- `b1,b2,tmp,ax`: block temporaries of size `n×b`.

Mathematical role
-----------------
The Chebyshev filter is applied via a block Clenshaw recurrence. These arrays
store the two previous block recurrences, a temporary block, and the block
operator application.
"""
struct POLFEDWorkspace
    b1::Matrix{Float64}
    b2::Matrix{Float64}
    tmp::Matrix{Float64}
    ax::Matrix{Float64}
end

"""
    POLFEDWorkspace(n,b)

Allocate a workspace for block size `b` in dimension `n`.
"""
POLFEDWorkspace(n::Int,b::Int)=POLFEDWorkspace(zeros(n,b),zeros(n,b),zeros(n,b),zeros(n,b))

"""
    apply_poly_block!(Y,A,Q,coeffs,smin,smax,work)

Apply the Chebyshev polynomial filter to a block `Q` for an explicit sparse A.

Output
------
Computes approximately

    Y = p(Â)Q,   Â=(A-cI)/d.

Method
------
Uses a block Clenshaw recurrence for stability:

    b_{K+1}=b_{K+2}=0,
    b_k = α_k Q + 2Â b_{k+1} - b_{k+2},
    p(Â)Q = α_0 Q + Â b_1 - b_2.
"""
function apply_poly_block!(Y::AbstractMatrix{Float64},A::SparseMatrixCSC{Float64,Int},Q::AbstractMatrix{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,work::POLFEDWorkspace)
    # Apply Y = p(Â)Q by block Clenshaw recurrence, where Â=(A-cI)/d.
    # Recurrence:
    #   B_{K+1}=B_{K+2}=0
    #   B_k = α_k Q + 2 Â B_{k+1} - B_{k+2},   k=K,...,1
    #   Y   = α₀ Q + Â B₁ - B₂
    #   Dominant cost: repeated block applications of A.
    K=length(coeffs)-1
    c=0.5*(smin+smax)
    invd=2.0/(smax-smin)
    α=2.0*invd
    β=-2.0*c*invd
    B1=work.b1
    B2=work.b2
    TMP=work.tmp
    AX=work.ax
    n,b=size(Q)
    fill!(B1,0.0)
    fill!(B2,0.0)
    @inbounds for k=(K+1):-1:2
        mul!(AX,A,B1)
        ck=coeffs[k]
        for j in 1:b
            @simd for i in 1:n
                TMP[i,j]=ck*Q[i,j]+α*AX[i,j]+β*B1[i,j]-B2[i,j]
            end
        end
        B1,B2,TMP=TMP,B1,B2
    end
    mul!(AX,A,B1)
    c0=coeffs[1]
    @inbounds for j in 1:b
        @simd for i in 1:n
            Y[i,j]=c0*Q[i,j]+invd*AX[i,j]-c*invd*B1[i,j]-B2[i,j]
        end
    end
    return Y
end

"""
    apply_poly_block!(Y,Aop,Q,coeffs,smin,smax,work)

Apply the Chebyshev polynomial filter to a block `Q` for a matrix-free operator. This is the generic version where the operator is accessed only through `apply_A_block!`.
"""
function apply_poly_block!(Y::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,Q::AbstractMatrix{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,work::POLFEDWorkspace)
    # Apply Y = p(Â)Q by block Clenshaw recurrence, where Â=(A-cI)/d.
    # Recurrence:
    #   B_{K+1}=B_{K+2}=0
    #   B_k = α_k Q + 2 Â B_{k+1} - B_{k+2},   k=K,...,1
    #   Y   = α₀ Q + Â B₁ - B₂
    #   Dominant cost: repeated block applications of A.
    K=length(coeffs)-1
    c=0.5*(smin+smax)
    invd=2.0/(smax-smin)
    α=2.0*invd
    β=-2.0*c*invd
    B1=work.b1
    B2=work.b2
    TMP=work.tmp
    AX=work.ax
    n,b=size(Q)
    fill!(B1,0.0)
    fill!(B2,0.0)
    @inbounds for k=(K+1):-1:2
        apply_A_block!(AX,A,B1)
        ck=coeffs[k]
        for j in 1:b
            @simd for i in 1:n
                TMP[i,j]=ck*Q[i,j]+α*AX[i,j]+β*B1[i,j]-B2[i,j]
            end
        end
        B1,B2,TMP=TMP,B1,B2
    end
    apply_A_block!(AX,A,B1)
    c0=coeffs[1]
    @inbounds for j in 1:b
        @simd for i in 1:n
            Y[i,j]=c0*Q[i,j]+invd*AX[i,j]-c*invd*B1[i,j]-B2[i,j]
        end
    end
    return Y
end

"""
    orth_block_polfed!(Q;rtol=1e-12)

Return an orthonormal basis for the columns of `Q`, truncating nearly dependent
directions using the diagonal of the QR R-factor.

Criterion
---------
If `R=QR` and `r0=max|R_ii|`, keep only columns with

    |R_ii| > rtol*r0.

This acts as a numerical rank-revealing QR truncation.
"""
function orth_block_polfed!(Q::AbstractMatrix{Float64};rtol::Float64=1e-12)
    # ------------------------------------------------------------
    # Orthonormalize and truncate a block of vectors
    # ------------------------------------------------------------
    #
    # Goal:
    #   Produce an orthonormal basis of the column space of Q,
    #   while removing numerically dependent directions.
    #
    #   1) Compute QR factorization:
    #        Q = Q_full R
    #
    #   2) Determine numerical rank:
    #        let d_i = |R_ii|
    #        let r0 = max_i d_i
    #        keep indices with d_i > rtol * r0
    #
    #   3) Truncate:
    #        Q ← first r columns of Q_full
    #
    #   4) If no directions survive:
    #        return empty matrix
    # ------------------------------------------------------------
    F=qr!(Q)
    R=Matrix(F.R)
    k=min(size(R,1),size(R,2))
    k==0 && return Matrix{Float64}(undef,size(Q,1),0)
    dr=abs.(diag(R[1:k,1:k]))
    r0=maximum(dr)
    r0==0 && return Matrix{Float64}(undef,size(Q,1),0)
    r=count(>(rtol*r0),dr)
    r==0 && return Matrix{Float64}(undef,size(Q,1),0)
    return Matrix(F.Q[:,1:r])
end

"""
    orth_against!(W,Q)

Project block `W` orthogonally against the column space of `Q`:

    W ← W - Q(Q^T W).

This is the standard orthogonal projector used in block orthogonalization.
"""
function orth_against!(W::AbstractMatrix{Float64},Q::AbstractMatrix{Float64})
    size(Q,2)==0 && return W
    W.-=Q*(transpose(Q)*W)
    return W
end

"""
    reorth_full!(W,Qhist,Qcur;passes=1)

Reorthogonalize `W` against all previously stored blocks in `Qhist` and the
current block `Qcur`.

Role in the algorithm
---------------------
Because filtered Krylov bases can lose orthogonality numerically, this step
stabilizes the reduced recurrence and keeps the assembled reduced matrix close
to symmetric block tridiagonal structure.
"""
function reorth_full!(W::AbstractMatrix{Float64},Qhist::Vector{Matrix{Float64}},Qcur::AbstractMatrix{Float64};passes::Int=1)
    for _ in 1:passes
        for Q in Qhist
            orth_against!(W,Q)
        end
        orth_against!(W,Qcur)
    end
    return W
end

"""
    block_qr_factor(W;rtol=1e-12)

Factor `W ≈ Q*B`, where `Q` has orthonormal columns and `B` is the truncated
upper factor.

Interpretation
--------------
This is the block analogue of normalizing the next Lanczos vector. The output
dimension determines the numerical rank of the next block.
"""
function block_qr_factor(W::AbstractMatrix{Float64};rtol::Float64=1e-12)
    # ------------------------------------------------------------
    # Truncated block QR factorization
    # ------------------------------------------------------------
    #
    # Input:
    #   W = candidate next block produced by filtered recurrence
    #
    # Goal:
    #   Factor W into
    #
    #       W ≈ Q B
    #
    #   where Q has orthonormal columns and only numerically significant
    #   directions are retained.
    #
    #   1) Compute QR factorization:
    #        W = Q_full R
    #
    #   2) Determine numerical rank from diagonal of R:
    #        let d_i = |R_ii|
    #        let r0 = max_i d_i
    #        keep indices with d_i > rtol * r0
    #
    #   3) Truncate:
    #        Q ← first r significant columns of Q_full
    #        B ← first r rows of R
    #
    #   4) If no significant directions remain:
    #        return empty factors
    # ------------------------------------------------------------
    F=qr!(W)
    R=Matrix(F.R)
    k=min(size(R,1),size(R,2))
    k==0 && return Matrix{Float64}(undef,size(W,1),0),Matrix{Float64}(undef,0,size(W,2))
    dr=abs.(diag(R[1:k,1:k]))
    r0=maximum(dr)
    r0==0 && return Matrix{Float64}(undef,size(W,1),0),Matrix{Float64}(undef,0,size(W,2))
    r=count(>(rtol*r0),dr)
    r==0 && return Matrix{Float64}(undef,size(W,1),0),Matrix{Float64}(undef,0,size(W,2))
    return Matrix(F.Q[:,1:r]),Matrix(R[1:r,1:size(W,2)])
end

"""
    assemble_T(Ablocks,Bblocks)

Assemble the reduced symmetric block tridiagonal matrix

    T = tridiag(A_1,...,A_m; B_1,...,B_{m-1})

from diagonal blocks `Ablocks` and off-diagonal blocks `Bblocks`.

Mathematical role
-----------------
This is the reduced representation of the filtered operator in the computed
block basis.
"""
function assemble_T(Ablocks::Vector{Matrix{Float64}},Bblocks::Vector{Matrix{Float64}})
    # ------------------------------------------------------------
    # Assemble reduced block-tridiagonal matrix T
    # ------------------------------------------------------------
    #
    # Input:
    #   Ablocks = diagonal blocks A₁,...,A_m
    #   Bblocks = off-diagonal blocks B₁,...,B_{m-1}
    #
    # Structure:
    #
    #        [ A₁   B₁ᵀ   0    ...        ]
    #        [ B₁   A₂   B₂ᵀ   ...        ]
    #   T =  [ 0    B₂   A₃    ...        ]
    #        [ ...                  ...   ]
    #
    #   1) Determine block sizes:
    #        r_j = size(A_j,1)
    #        total size N = Σ r_j
    #
    #   2) Initialize full dense matrix:
    #        T ← zeros(N,N)
    #
    #   3) Loop over blocks:
    #        for j = 1,...,m
    #            insert diagonal block:
    #                T[I_j,I_j] ← A_j
    #
    #            if j < m:
    #                insert off-diagonal blocks:
    #                    T[I_{j+1},I_j] ← B_j
    #                    T[I_j,I_{j+1}] ← B_jᵀ
    #
    #   4) Return symmetric matrix T
    # ------------------------------------------------------------
    nb=length(Ablocks)
    dims=[size(A,1) for A in Ablocks]
    N=sum(dims)
    T=zeros(Float64,N,N)
    o=1
    for j in 1:nb
        r=dims[j]
        I=o:o+r-1
        T[I,I].=Ablocks[j]
        if j<nb
            B=Bblocks[j]
            J=(o+r):(o+r+size(B,1)-1)
            T[J,I].=B
            T[I,J].=transpose(B)
        end
        o+=r
    end
    return Symmetric(T)
end

"""
    build_X!(X,Qhist,U)

Reconstruct Ritz vectors

    X = VU,   V=[Q_1 ... Q_m],

without explicitly forming `V` as a single large matrix.

Here `U` contains eigenvectors of the reduced matrix T.
"""
function build_X!(X::AbstractMatrix{Float64},Qhist::Vector{Matrix{Float64}},U::AbstractMatrix{Float64})
    fill!(X,0.0)
    o=1
    for Q in Qhist
        r=size(Q,2)
        X.+=Q*view(U,o:o+r-1,:)
        o+=r
    end
    return X
end

"""
    rayleigh!(λ,res,AX,A,X)

Compute original Rayleigh quotients and residual norms for explicit sparse A.

For each column `x_j` of X, computes

    λ_j = x_j^T A x_j,
    res_j = ||A x_j - λ_j x_j||_2.

These are the physically meaningful convergence diagnostics of the original
eigenproblem, not just of the reduced problem.
"""
function rayleigh!(λ,res,AX::AbstractMatrix{Float64},A::SparseMatrixCSC{Float64,Int},X::AbstractMatrix{Float64})
    apply_A_block!(AX,A,X)
    k=size(X,2)
    @inbounds for j in 1:k
        x=view(X,:,j)
        ax=view(AX,:,j)
        λj=dot(x,ax)
        λ[j]=λj
        s=0.0
        @simd for i in eachindex(x)
            r=ax[i]-λj*x[i]
            s+=r*r
        end
        res[j]=sqrt(s)
    end
    return λ,res
end

"""
    rayleigh!(λ,res,AX,Aop,X)

Compute original Rayleigh quotients and residual norms for a matrix-free operator.
"""
function rayleigh!(λ,res,AX::AbstractMatrix{Float64},A::MatrixFreeRealSymOp,X::AbstractMatrix{Float64})
    apply_A_block!(AX,A,X)
    k=size(X,2)
    @inbounds for j in 1:k
        x=view(X,:,j)
        ax=view(AX,:,j)
        λj=dot(x,ax)
        λ[j]=λj
        s=0.0
        @simd for i in eachindex(x)
            r=ax[i]-λj*x[i]
            s+=r*r
        end
        res[j]=sqrt(s)
    end
    return λ,res
end

"""
    refine_original_rr(A,X)

Perform an original-space Rayleigh–Ritz refinement on the candidate vectors X. This often improves the quality of the final vectors after interval extraction.
"""
function refine_original_rr(A,X)
    # ------------------------------------------------------------
    # Original-space Rayleigh–Ritz refinement
    # ------------------------------------------------------------
    #
    # Input:
    #   X = candidate approximate eigenvectors
    #
    #   1) Orthonormalize input:
    #        Q ← orth(X)
    #
    #   2) If Q is empty:
    #        return empty result
    #
    #   3) Project operator to subspace:
    #        H = Qᵀ A Q
    #
    #   4) Solve reduced dense eigenproblem:
    #        H y_j = μ_j y_j
    #
    #   5) Reconstruct refined vectors:
    #        Y = Q y_j
    #
    #   6) Compute original Rayleigh quotients and residuals:
    #        λ_j = y_jᵀ A y_j
    #        r_j = ||A y_j - λ_j y_j||
    #
    #   7) Sort by eigenvalue
    # ------------------------------------------------------------
    Q=orth_block_polfed!(copy(X))
    k=size(Q,2)
    k==0 && return Float64[],Matrix{Float64}(undef,size(X,1),0),Float64[]
    AQ=zeros(size(Q))
    apply_A_block!(AQ,A,Q)
    H=Symmetric(0.5*(transpose(Q)*AQ+transpose(AQ)*Q))
    E=eigen(H)
    Y=Q*E.vectors
    AY=zeros(size(Y))
    λ=zeros(k)
    res=zeros(k)
    rayleigh!(λ,res,AY,A,Y)
    p=sortperm(λ)
    return λ[p],Y[:,p],res[p]
end

"""
    extract_pairs(Qhist,Ablocks,Bblocks,A;nev,emin,emax,overextract=2,refine=true,extra_keep=8,chunk=64)

Extract approximate eigenpairs of the original problem from the reduced basis.

Returns
-------
`(λf,Xf,rf,θsel)` where `θsel` are the selected reduced eigenvalues before
original-space filtering.
"""
function extract_pairs(Qhist::Vector{Matrix{Float64}},Ablocks::Vector{Matrix{Float64}},Bblocks::Vector{Matrix{Float64}},A;nev::Int,emin::Float64,emax::Float64,overextract::Int=2,refine::Bool=true,extra_keep::Int=8,chunk::Int=64)
    # ------------------------------------------------------------
    # Direct extraction of approximate eigenpairs from reduced basis
    # ------------------------------------------------------------
    # Input:
    #   Qhist   = stored basis blocks Q₁,...,Q_m
    #   Ablocks,Bblocks = reduced block-tridiagonal data defining T
    #   A       = original operator

    #   1) Assemble reduced matrix:
    #        T ← block_tridiag(Ablocks,Bblocks)
    #
    #   2) Solve reduced eigenproblem:
    #        T u_j = θ_j u_j
    #
    #   3) Select leading reduced eigenvectors:
    #        keep m ≈ max(overextract*nev, nev+extra_keep)
    #        choose eigenvectors corresponding to largest θ_j
    #
    #   4) Reconstruct Ritz vectors in original space:
    #        X ← V U_sel,   where V=[Q₁,...,Q_m]
    #
    #   5) Evaluate original Rayleigh quotients / residuals:
    #        λ_j = x_jᵀ A x_j
    #        r_j = ||A x_j - λ_j x_j||
    #
    #   6) Interval filtering:
    #        keep only pairs with λ_j ∈ [emin,emax]
    #
    #   7) Optional original-space refinement:
    #        perform Rayleigh–Ritz on span{X_kept}
    #        recompute λ_j, r_j
    #        re-apply interval filtering
    #
    #   8) Residual-based pruning:
    #        sort surviving pairs by residual
    #        keep best nev+extra_keep
    #
    #   9) Final ordering:
    #        sort kept pairs by eigenvalue
    # ------------------------------------------------------------
    T=assemble_T(Ablocks,Bblocks)
    E=eigen(T)
    m=min(length(E.values),max(overextract*nev,nev+extra_keep))
    p=sortperm(E.values;rev=true)[1:m]
    θsel=E.values[p]
    Usel=E.vectors[:,p]
    n=size(Qhist[1],1)
    λ_keep=Float64[]
    res_keep=Float64[]
    idx_keep=Int[]
    for j0 in 1:chunk:m
        j1=min(j0+chunk-1,m)
        bs=j1-j0+1
        Uchunk=@view Usel[:,j0:j1]
        Xchunk=zeros(Float64,n,bs)
        AXchunk=zeros(Float64,n,bs)
        λchunk=zeros(Float64,bs)
        reschunk=zeros(Float64,bs)
        build_X!(Xchunk,Qhist,Uchunk)
        rayleigh!(λchunk,reschunk,AXchunk,A,Xchunk)
        @inbounds for j in 1:bs
            if emin<=λchunk[j]<=emax
                push!(λ_keep,λchunk[j])
                push!(res_keep,reschunk[j])
                push!(idx_keep,j0+j-1)
            end
        end
    end
    isempty(idx_keep) && return Float64[],Matrix{Float64}(undef,n,0),Float64[],θsel
    npre=min(length(res_keep),nev+extra_keep)
    ps0=sortperm(res_keep)[1:npre]
    λc=λ_keep[ps0]
    resc=res_keep[ps0]
    idxc=idx_keep[ps0]
    Xc=zeros(Float64,n,npre)
    build_X!(Xc,Qhist,@view Usel[:,idxc])
    if refine
        λc,Xc,resc=refine_original_rr(A,Xc)
        inside2=findall(i->emin<=λc[i]<=emax,1:length(λc))
        isempty(inside2) && return Float64[],Matrix{Float64}(undef,n,0),Float64[],θsel
        λc=λc[inside2]
        resc=resc[inside2]
        Xc=Xc[:,inside2]
    end
    nkeep=min(length(resc),nev+extra_keep)
    ps=sortperm(resc)[1:nkeep]
    λf=λc[ps]
    Xf=Xc[:,ps]
    rf=resc[ps]
    pe=sortperm(λf)
    return λf[pe],Xf[:,pe],rf[pe],θsel
end

"""
    _progress_line(m,λ,res)

Print a compact convergence summary for the one-cycle run.
"""
function _progress_line(m,λ,res)
    if isempty(res)
        println("m=",m,"  inside=0  maxres=Inf")
        return
    end
    rs=sort(res)
    r90=rs[clamp(round(Int,0.9*length(rs)),1,length(rs))]
    r95=rs[clamp(round(Int,0.95*length(rs)),1,length(rs))]
    println("m=",m,"  inside=",length(λ),"  r90=",r90,"  r95=",r95,"  maxres=",rs[end])
end

"""
    build_full_basis!(Qhist,Ablocks,Bblocks,A,Q1,coeffs,smin,smax,Wbuf,work;
                            mmax,threaded=true,rtol=1e-12,reorth_passes=1,show_progress=true)

Build a single full filtered block basis of length at most `mmax`.

Recurrence
----------
Starting from `Q1`, generate blocks according to a block Lanczos-like recurrence
for the filtered operator `p(Â)`:

    W = p(Â)Q_j - Q_{j-1}B_{j-1}^T
    A_j = sym(Q_j^T W)
    W ← W - Q_j A_j
    reorthogonalize W
    W = Q_{j+1} B_j

The basis blocks are stored in `Qhist`, and the reduced block tridiagonal data
are stored in `Ablocks` and `Bblocks`.
"""
function build_full_basis!(Qhist::Vector{Matrix{Float64}},Ablocks::Vector{Matrix{Float64}},Bblocks::Vector{Matrix{Float64}},A,Q1::Matrix{Float64},coeffs::AbstractVector{Float64},smin::Float64,smax::Float64,Wbuf::AbstractMatrix{Float64},work::POLFEDWorkspace;mmax::Int,threaded::Bool=true,rtol::Float64=1e-12,reorth_passes::Int=1,show_progress::Bool=true)
    empty!(Qhist) # clear stored basis blocks Q₁,...,Q_m from previous run
    empty!(Ablocks) # clear reduced diagonal blocks A_j
    empty!(Bblocks) # clear reduced off-diagonal coupling blocks B_j
    n=size(Q1,1) # dimension of the full operator
    Q=copy(Q1)  # current orthonormal block Q_j
    Qprev=Matrix{Float64}(undef,n,0) # previous block Q_{j-1}; empty at first step
    Bprev=Matrix{Float64}(undef,0,0) # previous coupling block B_{j-1}; empty at first step
    prog=show_progress ? Progress(mmax,"Cycle") : nothing
    # ------------------------------------------------------------
    # Block filtered Lanczos 
    # ------------------------------------------------------------
    # Given:
    #   Q₁  initial orthonormal block
    #   p(Â) Chebyshev polynomial filter
    #
    # For j = 1,2,...,mmax:
    #
    #   1) Apply filter:
    #        W ← p(Â) Q_j
    #
    #   2) Three-term recurrence (remove previous direction):
    #        W ← W - Q_{j-1} B_{j-1}ᵀ        (if j > 1)
    #
    #   3) Project onto current block:
    #        A_j ← Q_jᵀ W
    #        W   ← W - Q_j A_j = W - Q_j Q_jᵀ W = (1 - Q_j Q_jᵀ) W
    #
    #   4) Reorthogonalize:
    #        W ← (I - Q_hist Q_histᵀ) W
    #
    #   5) Normalize next block:
    #        [Q_{j+1}, B_j] ← QR(W)   (with rank truncation)
    #
    #   6) Store reduced operators:
    #        append A_j, B_j
    #
    # Result:
    #   V_m = [Q₁,...,Q_m]
    #   T   = block tridiagonal from {A_j, B_j}
    #
    #   such that:
    #        p(A) V_m ≈ V_m T
    # ------------------------------------------------------------
    for it in 1:mmax
        push!(Qhist,copy(Q)) # store current block so that V_m=[Q₁,...,Q_m] can later reconstruct Ritz vectors
        r=size(Q,2)  # current block rank; may drop below s due to numerical rank truncation
        W=view(Wbuf,:,1:r)  # working view for W_j with matching current block width
        apply_poly_block!(W,A,Q,coeffs,smin,smax,work) # compute W_j ← p(Â)Q_j, where Â=(A-cI)/d and p is the Chebyshev filter
        # subtract previous recurrence term Q_{j-1} B_{j-1}ᵀ
        # this is the block-Lanczos three-term recurrence part
        if size(Qprev,2)>0 && size(Bprev,1)>0 && size(Bprev,2)>0
            W.-=Qprev*transpose(Bprev)
        end
        # form reduced diagonal block
        # A_j = sym(Q_jᵀ W_j) where explicit Symmetric suppresses roundoff asymmetry
        Aj=Matrix(Symmetric(0.5*(transpose(Q)*W+transpose(W)*Q)))
        push!(Ablocks,Aj) # append diagonal block to reduced block-tridiagonal matrix T
        # remove component along current block:
        # W_j ← W_j - Q_j A_j
        W.-=Q*Aj
        for _ in 1:reorth_passes
            # reorthogonalize against all previously stored basis blocks and current block (does have effect on residuals in finite precision!)
            reorth_full!(W,Qhist,Q;passes=1)
        end
        # factor W ≈ Q_{j+1} B_j via truncated QR
        # Q_{j+1} becomes the next orthonormal basis block
        # B_j is the off-diagonal reduced coupling block
        Qnext,Bj=block_qr_factor(W;rtol=rtol)
        size(Qnext,2)==0 && break # stop if numerical rank collapsed completely; no further basis growth possible
        push!(Bblocks,Bj) # append coupling block B_j to reduced matrix data
        Qprev=Q # shift current block into previous-block slot for next recurrence step
        Bprev=Bj # shift current coupling block into previous-coupling slot
        Q=Qnext # advance recurrence: Q_j ← Q_{j+1}
        show_progress && next!(prog)
    end
    return nothing
end

"""
    polfed(A::MatrixFreeRealSymOp,nev,degree;kwargs...)

Polynomial-filtered block eigensolver for a matrix-free real symmetric operator.

Key parameters
--------------
- `nev`: desired number of eigenpairs in `[emin,emax]`.
- `s`: block size.
- `mmax`: number of block steps; reduced-space dimension is roughly `M≈s*mmax`.
- `degree`: polynomial degree of the Chebyshev filter.
- `smin`, `smax`: spectral bounds for the filter; required for matrix-free operation. 
- `sigma`: center of the filter; default is 0.0.
- `emin`, `emax`: interval for selecting Ritz pairs; default is entire real line.
- `tol`: tolerance for early exit based on residuals; default is 1e-10.
- `threaded`: whether to use multi-threading in the filter application; default is true.
- `overextract`: factor controlling how many Ritz pairs to extract from the reduced problem before original-space filtering; default is 2.
- `refine`: whether to perform original-space Rayleigh–Ritz refinement on the extracted Ritz vectors; default is true.
- `window_type`: type of window function for filter coefficients; default is `:step` (no window).
- `jackson`: whether to apply Jackson damping to the filter coefficients; default is true.
- `rtol`: relative tolerance for numerical rank truncation in block orthogonalization; default is 1e-12.
- `reorth_passes`: number of reorthogonalization passes to perform in each iteration; default is 1.
- `extra_keep`: number of extra Ritz pairs to keep beyond `nev` for safety in case of filtering inaccuracies; default is 8.
- `seed`: random seed for reproducibility of the initial block; default is 1.
- `show_progress`: whether to display a progress bar during basis construction; default is true.
- `chunk`: number of Ritz pairs to process at a time during extraction to manage memory usage; default is 64. This is 
"""
function polfed(A::MatrixFreeRealSymOp,nev::Int,degree::Int;s::Int=8,smin=nothing,smax=nothing,sigma::Float64=0.0,emin::Float64=-Inf,emax::Float64=Inf,tol::Float64=1e-10,threaded::Bool=true,overextract::Int=2,refine::Bool=true,window_type::Symbol=:step,jackson::Bool=true,mmax::Union{Nothing,Int}=nothing,rtol::Float64=1e-12,reorth_passes::Int=1,extra_keep::Int=8,seed::Int=1,show_progress::Bool=true,chunk=64)
    n=A.n
    (smin===nothing || smax===nothing) && error("Matrix-free POLFED requires spectral bounds smin and smax")
    coeffs=_build_coeffs(degree,smin,smax,sigma,emin,emax;window_type=window_type,jackson=jackson) # build chebyshev coefficients with optional jackson damping
    mmax===nothing && (mmax=ceil(Int,2.8*nev/s)) # lower bound for size of the working subspace (should realistically never be used)
    Random.seed!(seed)
    Q1=orth_block_polfed!(randn(n,s);rtol=rtol)# initial block Q₁ ∈ ℝ^{n×s}, orthonormalized random probe of invariant subspace
    work=POLFEDWorkspace(n,s) # workspace for Clenshaw recurrence (avoids allocations in filter application)
    Wbuf=zeros(Float64,n,s) # buffer for W = p(Â)Q_j and intermediate orthogonalization steps
    Qhist=Matrix{Float64}[] # stores block basis V = [Q₁,…,Q_m] (block-by-block)
    Ablocks=Matrix{Float64}[] # diagonal blocks A_j = Q_jᵀ p(Â) Q_j (reduced operator)
    Bblocks=Matrix{Float64}[] # off-diagonal blocks B_j from block Lanczos recurrence (coupling between blocks)
    # construct filtered block Krylov basis for p(Â): generates Q_j, A_j, B_j so that p(A) V ≈ V T
    build_full_basis!(Qhist,Ablocks,Bblocks,A,Q1,coeffs,smin,smax,Wbuf,work;mmax=mmax,threaded=threaded,rtol=rtol,reorth_passes=reorth_passes,show_progress=show_progress)
    # solve reduced eigenproblem T u = θ u,
    # reconstruct Ritz vectors X = V u,
    # compute original Rayleigh quotients λ and residuals,
    # keep candidates inside [emin,emax]
    λ,X,res,θ=extract_pairs(Qhist,Ablocks,Bblocks,A;nev=nev,emin=emin,emax=emax,overextract=overextract,refine=refine,extra_keep=extra_keep,chunk=chunk)
    _progress_line(length(Qhist),λ,res)
    if !isempty(res) && length(res)>=nev
        rs=sort(res)
        rs[nev] < tol && return λ[1:nev],X[:,1:nev],res[1:nev] # early exit if best nev residuals satisfy tolerance
    end
    if !isempty(res)
        p=sortperm(res)[1:min(nev,length(res))]
        return λ[p],X[:,p],res[p] # return nev best candidates sorted by residual (robust fallback)
    else
        return Float64[],Matrix{Float64}(undef,n,0),Float64[] # no candidates found inside interval
    end
end

"""
    polfed(matvec!, n, nev, degree; matblock!=nothing, kwargs...)

Convenience wrapper constructing a matrix-free real symmetric operator
from user-supplied actions and calling `polfed`.

Additional arguments
---------------------
- `matvec! :: Function`
    In-place vector action `y ← A*x`.

- `n :: Int`
    Dimension of the operator.

Additional keyword arguments
-----------------------------
- `matblock! :: Function or nothing` (default: `nothing`)
    Optional in-place block action `Y ← A*X`. This should always
    be added for faster performance when available.

- `kwargs...`
    Passed directly to the main `polfed(::MatrixFreeRealSymOp, ...)` routine.
"""
function polfed(matvec!::F,n::Int,nev::Int,degree::Int;matblock!::Union{Nothing,G}=nothing,kwargs...) where {F,G}
    Aop=isnothing(matblock!) ? MatrixFreeRealSymOp(n,matvec!) : MatrixFreeRealSymOp(n,matvec!,matblock!)
    return polfed(Aop,nev,degree;kwargs...)
end

########################################################
################### 1D LAPLACIAN TEST ##################
########################################################

function lap1d(n::Int)
    e=ones(Float64,n)
    return spdiagm(-1=>-e[1:end-1],0=>2e,1=>-e[1:end-1])
end
function lap1d_matvec!(y::AbstractVector{Float64},x::AbstractVector{Float64})
    n=length(x)
    @inbounds begin
        y[1]=2x[1]-x[2]
        for i in 2:n-1
            y[i]=-x[i-1]+2x[i]-x[i+1]
        end
        y[n]=-x[n-1]+2x[n]
    end
    return y
end
function lap1d_matblock!(Y::AbstractMatrix{Float64},X::AbstractMatrix{Float64}) # block action Y ← A*X for 1d Laplacian
    n,b=size(X)
    Threads.@threads for j in 1:b
        @inbounds begin
            Y[1,j]=2.0*X[1,j]-X[2,j]
            for i in 2:n-1
                Y[i,j]=-X[i-1,j]+2.0*X[i,j]-X[i+1,j]
            end
            Y[n,j]=-X[n-1,j]+2.0*X[n,j]
        end
    end
    return Y
end
function lap1d_exact_eigs(n::Int)
    [(2-2*cos(j*pi/(n+1))) for j in 1:n]
end
function lap1d_target_window(n::Int,nev::Int)
    λexact=lap1d_exact_eigs(n)
    mid=clamp(round(Int,n/2),1,n)
    i1=clamp(mid-fld(nev,2),1,n)
    i2=clamp(i1+nev-1,1,n)
    i1=clamp(i2-nev+1,1,n)
    emin=λexact[i1]
    emax=λexact[i2]
    sigma=0.5*(emin+emax)
    return λexact,i1,i2,emin,emax,sigma
end

if abspath(PROGRAM_FILE)==@__FILE__

########################################################
######################## TEST ##########################
########################################################

BLAS.set_num_threads(Sys.CPU_THREADS)
    
n=150_000
nev=1000
s=64
degree=400
b=3.0
mmax=ceil(Int,b*nev/s)
window_type=:smooth # for very large n use :delta here

λexact,i1,i2,emin,emax,sigma=lap1d_target_window(n,nev)

@info "\n matrix size n=$(n) \n nev=$(nev) \n s=$(s) \n degree=$(degree) \n b=$(b) \n mmax=$(mmax) \n window_type=$(window_type)"
Aop=MatrixFreeRealSymOp(n,lap1d_matvec!,lap1d_matblock!)

# smin, smax are the spectral bounds which for the case of the 1d laplacian are 0.0 - 4.0.
# emin and emax are the target energies in the middle and we want nev of them (this is what lap1d_target_window computes)
# good convergence is an interplay between degree and b. Usually a higher degree allows for a smaller b and vice versa. This depends on the sparse matrix and target window.
# also if convergence is just a bit smaller than wanted increase reorth_passes (usually 1 is enough)
λ,X,res=polfed(Aop,nev,degree;s=s,smin=0.0,smax=4.0,sigma=sigma,emin=emin,emax=emax,tol=1e-6,window_type=:delta,mmax=mmax,overextract=2,reorth_passes=1,show_progress=true,extra_keep=8,chunk=64)

end

#################
#### RESULTS ####
#################

#= What one should get for the laplacian 1d sparse matrix (M3 max, 64 Gb RAM)

nev_conv : number of reduced eigenvectors of T whose tail is decoupled (||B_next * u_tail|| small)
inside : number of Ritz pairs with Rayleigh quotients inside the target interval [emin,emax] (we wanted nev=1000)
r90 : 90th percentile of the original residual norms of the Ritz pairs inside the interval
r95 : 95th percentile of the original residual norms of the Ritz pairs inside the interval
maxres : maximum original residual norm among the Ritz pairs inside the interval

NOTE: This is complete overkill (b=5.0 and degrees really high). Realistically lower bounds should be found with more testing!
NOTE: There is a final dense eigen! that is not taken into the main timing and does not influence the runtime for large n at all, but will cause some RAM usage as it tries to verify the accuracy of the Ritz pairs.

┌ Info: 
│  matrix size n=10000 
│  nev=1000 
│  s=64 
│  degree=32 
│  b=5.0 
│  mmax=79 
└  window_type=smooth
Cycle 100%|█████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:19
m=79  inside=1000  r90=1.263385139348521e-14  r95=1.4465502435677196e-14  maxres=3.1117078311317e-14

┌ Info: 
│  matrix size n=60000 
│  nev=1000 
│  s=64 
│  degree=128 
│  b=3.0 
│  mmax=47 
└  window_type=smooth
Cycle 100%|████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:36
m=47  inside=999  r90=2.3431295846617787e-14  r95=2.8476247888023835e-14  maxres=2.175237376209596e-12

┌ Info: 
│  matrix size n=150000 
│  nev=1000 
│  s=64 
│  degree=400 
│  b=3.0 
│  mmax=47 
└  window_type=smooth
Cycle 100%|████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:02:13
m=47  inside=998  r90=3.370908181692764e-14  r95=4.0029332417106185e-14  maxres=7.502190770434304e-14

┌ Info: 
│  matrix size n=300000 
│  nev=1000 
│  s=64 
│  degree=750 
│  b=3.0 
│  mmax=47 
└  window_type=smooth
Cycle 100%|████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:06:56
m=47  inside=998  r90=4.851085163074631e-14  r95=5.668566328697754e-14  maxres=9.447243092189946e-14

# Run with nev=1600 and degree=800 

 Info: 
│  matrix size n=350000 
│  nev=1600 
│  s=64 
│  degree=800 
│  b=3.0 
│  mmax=75 
└  window_type=delta
Cycle 100%|████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:14:51
m=75  inside=1599  r90=5.406490405565137e-14  r95=6.560545613539887e-14  maxres=2.12138039476538e-11

############ COMPARING n=450_000 with varying degree and b ############

┌ Info: 
│  matrix size n=450000 
│  nev=1000 
│  s=64 
│  degree=728 
│  b=5.0 
│  mmax=79 
└  window_type=smooth
Cycle 100%|█████████████████████████████████████████████████████████████████████████████████████████| Time: 0:23:38
m=79  inside=998  r90=6.294294088644106e-14  r95=7.424619128109143e-14  maxres=1.1966103393555824e-13

┌ Info: 
│  matrix size n=450000 
│  nev=1000 
│  s=64 
│  degree=1250 
│  b=3.0 
│  mmax=47 
└  window_type=smooth
Cycle 100%|████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:15:37
m=47  nev_conv=1847  inside=1007  r90=6.286578274842279e-14  r95=7.545085171066087e-14  maxres=0.01888119337120290  -> sligthly faster but includes some extra pairs (exactly 7) just outside the interval as inside and therefore increases maxres. maxres is only small if really there is no small outside polution.

┌ Info: 
│  matrix size n=650000 
│  nev=1000 
│  s=64 
│  degree=980 
│  b=5.0 
│  mmax=79 
└  window_type=smooth
Cycle 100%|█████████████████████████████████████████████████████████████████████████████████████████| Time: 0:38:09
m=79  inside=1000  r90=7.199055102796388e-14  r95=8.595247481961762e-14  maxres=1.2347500904341446e-13




######################################################################
For very large matrices it is extremely important to know how degree and b interplay!
Usually the approach is to increase degree rather than increase b.

┌ Info: 
│  matrix size n=1048576
│  nev=1000 
│  s=64 
│  degree=3500 
│  b=2.0 
│  mmax=32 
└  window_type=delta
Cycle 100%|████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:58:11
m=32  inside=1000  r90=8.773094039016345e-14  r95=1.0435955357749034e-13  maxres=2.070560874529873e-13

┌ Info: 
│  matrix size n=1048576 
│  s=64 
│  degree=2500 
│  b=3.0
│  mmax=47 
└  window_type=delta
Cycle 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 2:13:13
m=47  inside=999  r90=6.2752806221874154e-15  r95=6.709334573082644e-15  maxres=2.576533337148389e-11

┌ Info: 
│  matrix size n=1048576 
│  s=64 
│  degree=1600 
│  b=5.0
│  mmax=94 
└  window_type=delta
Cycle 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 3:55:04
m=94  inside=999  r90=5.991394217652792e-15  r95=6.679527965337017e-15  maxres=2.4792930459694443e-11

=#

