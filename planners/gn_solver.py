# Lint as: python3
"""Second order generic path optimization.

Optimizes path residual defined by pairwise function r^t(x^t, x^{t+1}).
"""

import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

FLOAT = tf.float64

# Factorize block tridiagonal symmetric positive definite matrix.
# See: https://software.intel.com/en-us/node/531896
# D: array of T [N x dim x dim] s.p.d. blocks for the main diagonal
# B: array of (T-1) [N x dim x dim] blocks for the lower diagonal (upper diagonal is its transpose)
def factorize(D, B):
  T = len(D)
  L = [None] * T #tf.TensorArray(tf.float32, T)
  C = [None] * T #tf.TensorArray(tf.float32, T)
  L[0] = tf.linalg.cholesky(D[0])
  for t in range(T-1):
    # solve c = b * L^{-T}
    b_tr = tf.transpose(B[t], [0, 2, 1])
    c_tr = tf.linalg.triangular_solve(L[t], b_tr)
    c = tf.transpose(c_tr, [0, 2, 1])
    l = D[t+1] - tf.matmul(c, c, transpose_b = True)
    C[t] = c
    L[t+1] = tf.linalg.cholesky(l)
  return L, C

# Solve factorized block tridiagonal linear system.
# See: https://software.intel.com/en-us/node/531897
# L: array of T [N x dim x dim] lower triangular blocks for the main diagonal
# C: array of (T-1) [N x dim x dim] blocks for the lower diagonal (upper diagonal is its transpose)
# F: array of T [N x dim x K] blocks for the system r.h.s
def solve_factorized(L, C, F):
  T = len(L)
  # solve linear system for lower bi-diagonal
  Y = [None] * T
  Y[0] = tf.linalg.triangular_solve(L[0], F[0])
  for t in range(1, T):
    G = F[t] - tf.matmul(C[t-1], Y[t-1])
    Y[t] = tf.linalg.triangular_solve(L[t], G)
  # solve linear system for upper bi-diagonal
  X = [None] * T
  X[T-1] = tf.linalg.triangular_solve(L[T-1], Y[T-1], adjoint=True)
  for t in range(T-2, -1, -1):
    # algorithm description says to use F[t] instead of Y[t] here, but
    # that looks to be an error
    Z = Y[t] - tf.matmul(C[t], X[t+1], transpose_a=True)
    X[t] = tf.linalg.triangular_solve(L[t], Z, adjoint=True)
  return X

# Solve block tridiagonal linear system.
# See: https://software.intel.com/en-us/node/531897
# D: array of T [N x dim x dim] s.p.d. blocks for the main diagonal
# B: array of (T-1) [N x dim x dim] blocks for the lower diagonal (upper diagonal is its transpose)
# F: array of T [N x dim x K] blocks for the system r.h.s
# @tf.function
def solve_block_tridiagonal(D, B, F):
  L, C = factorize(D, B)
  X = solve_factorized(L, C, F)
  return X

# Build Gauss-Newton matrix from Jacobians. If J is a matrix with J_curr in main
# diagonal and J_prev in lower diagonal, this function returns 2 J^T J.
# J_curr: array of T [N x dim_r x dim_x] blocks of dr^t/dx^t
# J_prev: array of (T-1) [N x dim_r x dim_x] blocks of dr^t/dx^{t-1}
# damping: weight of the added main diagonal to ensure positive definiteness
# returns: array of T [N x dim_x x dim_x] and array of (T-1) [N x dim_x x dim_x]
# which are the main and lower diagonals of the Gauss-Newton matrix.
def gauss_newton_matrix(J_curr, J_prev, damping = 0.0):
  T = len(J_curr)
  dim_x = J_curr[0].shape[-1]
  D = [None] * T
  B = [None] * (T-1)
  I = damping * tf.eye(dim_x, batch_shape=[1], dtype=FLOAT)
  # fill main diagonal
  for t in range(T-1):
    D[t] = I + 2.0 * (tf.matmul(J_curr[t], J_curr[t], transpose_a=True) + \
       tf.matmul(J_prev[t], J_prev[t], transpose_a=True))
  D[T-1] = I + 2.0 * tf.matmul(J_curr[T-1], J_curr[T-1], transpose_a=True)
  # fill lower diagonal
  for t in range(T-1):
    B[t] = 2.0 * tf.matmul(J_curr[t+1], J_prev[t], transpose_a=True)
  return D, B


# Build gradient of sum of squared residuals from residuals and their Jacobians.
# If J is a matrix with J_curr in main diagonal and J_prev in lower diagonal,
# this function returns 2 J^T r.
# r: array of T [N x dim_r] residuals r(x^{t-1}, x^t)
# J_curr: array of T [N x dim_r x dim_x] blocks of dr^t/dx^t
# J_prev: array of (T-1) [N x dim_r x dim_x] blocks of dr^t/dx^{t-1}
# returns: array of T [N x dim_x] gradients
def gradient_vector(r, J_curr, J_prev):
  T = len(r)
  g = [None] * T
  for t in range(T-1):
    g[t] = 2.0 * (tf.matmul(J_curr[t], tf.expand_dims(r[t], -1), transpose_a=True) + \
      tf.matmul(J_prev[t], tf.expand_dims(r[t+1], -1), transpose_a=True))[...,0]
  g[T-1] = 2.0 * tf.matmul(J_curr[T-1], tf.expand_dims(r[T-1], -1), transpose_a=True)[...,0]
  return g


# Build a Gaussian noise block vector given lower-triangular factorization of
# the covariance matrix.
# L: array of T [N x dim x dim] lower triangular blocks for the main diagonal
# C: array of (T-1) [N x dim x dim] blocks for the lower diagonal
# returns: array of T [N x dim_x x 1] noise vector
def block_random_normal(L, C, num_samples=1):
  T = len(L)
  N = tf.shape(L[0])[0]
  dim_x = L[0].shape[-1]
  U = tf.split(tf.random.normal([N, dim_x, T*num_samples]), T, axis=-1)
  return solve_factorized(L, C, U)

# D: array of T [N x dim x dim] s.p.d. blocks for the main diagonal
# B: array of (T-1) [N x dim x dim] blocks for the lower diagonal (upper diagonal is its transpose)
# B_upper: array of (T-1) [N x dim x dim] blocks for the upper diagonal
def tridiagonal_to_dense(D, B, B_upper = None, fill_lower = True, fill_upper = True):
  T = len(D)
  N = D[0].shape[0]
  dim_r = D[0].shape[-2]
  dim_c = D[0].shape[-1]
  if B_upper is None and fill_upper:
    B_upper = [tf.transpose(B[t], [0,2,1]) for t in range(T-1)]
  M = np.zeros([N,T*dim_r,T*dim_c])
  for t in range(T):
    # main diagonal
    M[:,dim_r*t:dim_r*(t+1), dim_c*t:dim_c*(t+1)] = D[t]
  for t in range(T-1):
    # lower diagonal
    if fill_lower:
      M[:,dim_r*(t+1):dim_r*(t+2), dim_c*t:dim_c*(t+1)] = B[t]
    # upper diagonal
    if fill_upper:
      M[:,dim_r*t:dim_r*(t+1), dim_c*(t+1):dim_c*(t+2)] = B_upper[t]
  return M.astype(np.float32)

# r: array of T [N x dim_r] residuals r(x^{t-1, x^t)
# J_curr: array of T [N x dim_r x dim_x] Jacobian blocks of dr^t/dx^t
# J_prev: array of (T-1) [N x dim_r x dim_x] Jacobian blocks of dr^t/dx^{t-1}
# damping: scalar weight of the added main diagonal to ensure positive definiteness
# noise_amount: None or scalar amount to add Gaussian noise with covariance
# defined by the Gauss-Newton matrix (see: Riemannian Langevin dynamics).
# returns: array of T [N x dim_x] that minimizes sum of squared residuals
def gauss_newton_solve(r, J_curr, J_prev, MJ_curr, MJ_prev, damping = 0.0, noise_amount = None):
  T = len(r)
  # build gradient
  g = gradient_vector(r, J_curr, J_prev)
  for t in range(T):
    g[t] = tf.expand_dims(g[t], -1) # add dimension for solver compatibility
  # build hessian
  D, B = gauss_newton_matrix(MJ_curr, MJ_prev, damping)
  # solve system
  L, C = factorize(D, B)
  X = solve_factorized(L, C, g)
  # print(f"Gauss-Newton solve total: {t1-t0}")
  for t in range(T):
    X[t] = X[t][..., 0] # remove unused dimension
  # add noise to solution (optional)
  if not (noise_amount is None):
    X_noise = block_random_normal(L, C)
    for t in range(T):
      # TODO(imordatch): do we need to multiply by 2 to be strictly correct?
      X[t] += noise_amount * X_noise[t][..., 0]
  return X


# Calculate value and Jacobian of vector function at x
# x: [N x (...)]
# output: [N x (...) x (...)]
def jacobian(function, x):
  with tf.GradientTape() as g:
    g.watch(x)
    f = function(x)
  f_x = g.batch_jacobian(f, x)
  return f, f_x


# Calculate first and second derivatives of a scalar function at x
# x: [N x ...]
# returns:
# f_x: [N x ... x dim_x]
# f_xx: [N x ... x dim_x]
def scalar_derivatives(function, x):
  with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
      gg.watch(x)
      f = function(x)
    f_x = gg.gradient(f, x)
  f_xx = g.gradient(f_x, x)
  return f_x, f_xx


# Construct residual and Jacobian blocks for a function at x.
# pair_residual_func: function mapping ([N x dim_x], [N x dim_x]) tuple to
# [N x dim_r] residual vector
# init_residual_func: function mapping [N x dim_x] to
# [N x dim_r] residual vector of initial conditions
# x: [N x T x dim_x] function input chain
# pair_loss_func: function mapping [N x dim_r] to [N x 1] losses
# (when None, L2 loss is used)
# returns: T array of [N x dim_r] residuals and two [N x dim_r x dim_x]
# Jacobians dr^t/dx^t and dr^t/dx^{t-1}
def make_blocks(pair_residual_func, init_residual_func, x,
                pair_loss_func = None):
  dim_x = x.shape[-1]
  T = x.shape[1]
  # collect batch inputs and wrap function
  X_prev = tf.reshape(x[:,:-1,:], [-1, dim_x]) # [N*(T-1) x dim_x]
  X_curr = tf.reshape(x[:,+1:,:], [-1, dim_x]) # [N*(T-1) x dim_x]
  X_pair = tf.stack([X_prev, X_curr], axis=1) # [N*(T-1) x 2 x dim_x]
  batch_pair_func = lambda x : pair_residual_func(x[:,0,:], x[:,1,:])

  # compute batch pair residuals and Jacobians
  R_pair, J_pair = jacobian(batch_pair_func, X_pair)
  dim_r = R_pair.shape[-1]
  R_pair = tf.reshape(R_pair, [-1, T - 1, dim_r])
  J_pair = tf.reshape(J_pair, [-1, T - 1, dim_r, 2, dim_x])
  R_pair = tf.cast(R_pair, FLOAT)
  J_pair = tf.cast(J_pair, FLOAT)
  # initial condition residuals and Jacobians
  R_init, J_init = jacobian(init_residual_func, x[:, 0, :])
  R_init = tf.cast(R_init, FLOAT)
  J_init = tf.cast(J_init, FLOAT)

  # incorporate loss derivaties into residuals and Jacobians
  if pair_loss_func:
    L_r, L_rr = scalar_derivatives(pair_loss_func, R_pair)
    # target adjustment is: G = J' * L_rr * J, g = J * L_r
    # this function outputs J and r only, such that G = J'*J, g = J*r
    # so we can compensate:
    R_pair = 0.5 * L_r# / tf.sqrt(0.5 * L_rr)
    L_rr = tf.reshape(L_rr, [-1, T-1, dim_r, 1])
    # J_pair = J_pair * tf.sqrt(0.5 * L_rr)
    M = tf.sqrt(0.5 * L_rr)
  else:
    M = None

  # split into arrays along time axis
  r = [None] * T
  J_curr = [None] * T
  J_prev = [None] * (T-1)
  r[0] = R_init
  J_curr[0] = J_init
  for t in range(T-1):
    r[t+1] = R_pair[:,t,:]
    J_curr[t+1] = J_pair[:,t,:,1,:]
    J_prev[t] = J_pair[:,t,:,0,:]

  if M is not None:
    MJ_curr = [None] * T
    MJ_prev = [None] * (T-1)
    MJ_curr[0] = J_init
    for t in range(T-1):
      MJ_curr[t+1] = M[:,t,:,:]*J_pair[:,t,:,1,:]
      MJ_prev[t] = M[:,t,:,:]*J_pair[:,t,:,0,:]
  else:
    MJ_curr = J_curr
    MJ_prev = J_prev

  return r, J_curr, J_prev, MJ_curr, MJ_prev

@tf.function
def solve_step(pair_residual_function, init_residual_function, x,
               pair_loss_func = None, damping=0.0, noise_amount=None):
  r, J_curr, J_prev, MJ_curr, MJ_prev = make_blocks(pair_residual_function,
                                  init_residual_function, x, pair_loss_func)
  dx = gauss_newton_solve(r, J_curr, J_prev, MJ_curr, MJ_prev, damping, noise_amount)
  dx = tf.cast(tf.stack(dx, axis=1), tf.float32)
  return x - dx
