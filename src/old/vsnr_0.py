import numpy as np
from numpy.fft import fft2, ifft2
from numpy import real
from numpy import conj


def vsnr(u0, eps, p, filters, alpha, maxit, prec, C,
         primal_var=None, dual_var=None):
    """
    This function helps removing "structured" additive noise. By
    "structured", we mean that the noise consists of convolving a white noise
    with a given filter.

    PD stands for Primal-Dual, as the core of the program is a first-order
    primal-dual algorithm (described in "A first-order Primal-Dual Algorithm
    for convex problems with application to imaging", by A. Chambolle and T.
    Pock).

    This function solves (in the sense that the duality gap is less than prec):
    Min_{Lambda} ||nabla u||_{1,eps} + sum_{i=1}^m alpha(i)||Lambda_i||_p_i
    over the constraint that ||lambda||_infty<=C

    where :
    - u=u0+sum_{i=1}^m conv(Lambda_i,Gabor(i))
    - ||q||_{1,eps}= sum_{i=1}^n f(|q|) where f(x)=|x| if |x|>eps and
    - f(x)=x^2/2eps+eps/2 otherwise.
    - ||.||_p is the standard l^p-norm

    Any comments: please contact Pierre Weiss, pierre.armand.weiss@gmail.com

    Parameters
    ----------
    u0: numpy.ndarray((m, n))
        Original image
    eps: float
        Regularization parameter for TV norm (can be 0)
    q:
        Either a value in {1,2,Inf} or a vector of size p with values
        in the previous set
    filters: numpy.ndarray((m, n, p))
        Array containing the shape of the p filters
    alpha:
        Either a positive value or a vector of size p with all positive values
    maxit: int
        Number of maximum iterations
    prec: float
        a positive value that specifies the desired precision (typically=1e-2)
    C:
        l-infinite constraint on lambda
    primal_var, dual_var:
        Optional parameters which are estimates of the primal and dual
        solutions. (Note : EstP is actually Lambda)

    Returns
    -------
    img_den: numpy.ndarray((m, n))
        Denoised image as : img_den = img + sum_{i=1}^p conv(Lambda_i,Gabor(i))
    gap, primal, dual:
        vectors of size equal to the iterations number that specifies the
        duality gap, Primal cost and Dual Cost at each iteration
    EstP, EstD:
        Solutions of primal and dual problem
    """
    shape = u0.shape
    assert (len(filters.shape) == 3)
    assert (shape[0] == filters.shape[0] and shape[1] == filters.shape[1])

    # initialization
    gaps = primals = duals = np.zeros(maxit)
    nfilters = filters.shape[2]
    alpha = alpha * np.ones(nfilters)
    p = p * np.ones(nfilters)

    if primal_var is None:
        lambda_ = np.zeros_like(filters)
    else:
        lambda_ = primal_var
    if dual_var is None:
        q = np.zeros((shape[0], shape[1], 2))
    else:
        q = dual_var

    # gradient calculation
    gu0 = np.zeros((shape[0], shape[1], 2))
    gu0[..., 0] = drond1(u0)
    gu0[..., 1] = drond2(u0)

    # filters fourier transform
    filters_fft = []
    for k in range(nfilters):
        filters_fft.append(fft2(filters[..., k]))

    # Primal-Dual Algorithm
    #######################

    # metric specification
    N = np.ones_like(u0)
    M = np.ones_like(lambda_)

    lambdab = lambda_.copy()

    # initial duality gap computation
    #################################
    b = np.zeros_like(u0)
    for k in range(nfilters):
        b += real(ifft2(fft2(lambda_[..., k]) * filters_fft[k]))
    # b = np.real(b)  # at this point, b represents the noise

    # Current denoised image estimation
    u = u0 + b

    # primal cost computation
    d1u = drond1(u)
    d2u = drond2(u)
    norm2 = d1u[0] ** 2 + d2u[1] ** 2
    if eps == 0:
        norm = np.sum(np.sqrt(norm2))
    else:
        norm = np.sum(np.minimum(norm2 / eps, np.sqrt(norm2)) -
                      0.5 * np.minimum(norm2 / eps, eps))
    c_lambda = 0.
    for k in range(nfilters):
        c_lambda += Phi(lambdab[..., k], p[k], alpha[k])
    primals[0] = norm + c_lambda

    # dual cost computation
    q_grad_fft = fft2(drond1T(q[..., 0]) + drond2T(q[..., 1]))
    Astarq = np.zeros_like(lambda_)
    for k in range(nfilters):
        Astarq[..., k] = real(ifft2(np.conj(filters_fft[k]) * q_grad_fft))

    # Gstar computation
    Gstar = 0.
    for k in range(nfilters):
        Gstar += PhiStar(-M[..., k] * Astarq[..., k], p[k], C, alpha[k])

    # Fstar comuptation ( F * (q) )
    norm = N * np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2)
    if np.max(norm) > 1:
        Fstar = np.inf
    else:
        Fstar = fnorm(norm ** 2 -
                      np.sum(np.sum(gu0[..., 0] * N * q[..., 0])) -
                      np.sum(np.sum(gu0[..., 1] * N * q[..., 1])))
    Fstar *= 0.5 * eps
    duals[0] = -Fstar - Gstar

    gaps[0] = primals[0] - duals[0]

    # A largest singular value calculation
    ######################################
    d1 = np.zeros_like(u0)
    d1[-1, 0] = 1
    d1[0, 0] = -1
    d2 = np.zeros_like(u0)
    d2[0, -1] = 1
    d2[0, 0] = -1

    d1h = np.fft.fft2(d1)
    d2h = np.fft.fft2(d2)

    H = np.zeros_like(u0)
    for k in range(nfilters):
        H += np.abs(filters_fft[k]) ** 2
    L = np.sqrt(np.max(H * (np.abs(d1h ** 2) + np.abs(d2h ** 2))))
    print('Operator norm :', L)

    gamma = np.min(alpha)
    weight = 1
    tau = weight / L
    sigma = 1 / (tau * L ** 2)
    theta = 1
    nit = 2

    gap = np.inf
    if np.isnan(gaps[0]):
        print(f"INITIAL DUAL GAP IS INFINITE - PROGRAM WILL STOP AFTER {maxit} "
              f"ITERATIONS \n")
        gaps[0] = 1e16

    # The actual algorithm
    #####################
    while nit < maxit and gap > prec * gaps[0]:
        # I.1/ q_{n+1}=(I+sigma partial F^*)^{-1}(q_n+sigma A lambdab_n)
        # Computation of the convolutions with lambdab_n

        b = np.zeros_like(u0)
        for k in range(nfilters):
            b += real(ifft2(fft2(lambdab[..., k]) * filters_fft[k]))

        # Current denoised image estimation
        u = u0 + b

        # Gradient (corresponds to tilde q_n in the article)
        q[..., 0] += sigma * drond1(u)
        q[..., 1] += sigma * drond2(u)

        # Resolvent operator...
        norm = np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2)
        vmax = np.maximum(N * norm, N * eps * sigma + 1)
        q[..., 0] /= vmax
        q[..., 1] /= vmax

        # II.1/ lambda_{n+1}=(I+tau partial G)^{-1}(lambda_{n+1}-tau A^*q_{n+1})
        # ATq_{n+1} Computation
        q_grad_fft = fft2(drond1T(q[..., 0]) + drond2T(q[..., 1]))

        Astarq = np.zeros_like(lambda_)
        for k in range(nfilters):
            Astarq[..., k] = real(ifft2(conj(filters_fft[k]) * q_grad_fft))
        lambdau = lambda_ - tau * Astarq

        # resolvent of (I+tau partial G)^{-1} computation
        for k in range(nfilters):
            lambdau[..., k] = Prox_Phi(lambdau[..., k], p[k], M[..., k],
                                       tau, alpha[k], C)

        # III/ Step size update (TO BE DONE)
        if np.sum(p == 2) == len(p):  # If all phi_i are l2
            if eps > 0:
                mu = 2 * np.sqrt(gamma * eps) / L
                tau = mu / (2 * gamma)
                sigma = mu / (2 * eps)
                theta = 1 / (1 + mu)
            else:
                theta = 1 / np.sqrt(1 + 2 * gamma * tau);
                tau = theta * tau
                sigma = sigma / theta
        else:
            pass

        # IV / Correction bar x^{n + 1} = x^{n + 1} + theta(x^{n + 1} - x ^ n)
        lambdab = lambdau + theta * (lambdau - lambda_)
        lambda_ = lambdau

        # V/ Display (NOTE : computation of the cost function could be done in
        # here, i.e. only once in a while)

        if nit % 10 == 0:
            # I.2 / Computation of the primal cost(VALIDATED)
            d1u = drond1(u)
            d2u = drond2(u)
            norm2 = d1u ** 2 + d2u ** 2
            if eps == 0:
                norm = np.sum(np.sqrt(norm2))
            else:
                norm = np.sum(np.minimum(norm2 / eps, np.sqrt(norm2)) -
                              0.5 * np.minimum(norm2 / eps, eps))
            c_lambda = 0.
            for k in range(nfilters):
                c_lambda += Phi(lambdab[..., k], p[k], alpha[k])
            primals[nit] = norm + c_lambda

            # II.2/ Computation of the dual cost
            # Computation of Gstar=G*(-A*q)

            # Gstar computation
            Gstar = 0.
            for k in range(nfilters):
                Gstar += PhiStar(-M[..., k] * Astarq[..., k], p[k], C, alpha[k])

            # Fstar comuptation ( F * (q) )
            norm = N * np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2)
            if np.max(norm) > 1:
                Fstar = np.inf
            else:
                Fstar = fnorm(norm ** 2 -
                              np.sum(np.sum(gu0[..., 0] * N * q[..., 0])) -
                              np.sum(np.sum(gu0[..., 1] * N * q[..., 1])))
                Fstar *= eps / 2

            duals[nit] = -Fstar - Gstar
            gaps[nit] = primals[nit] - duals[nit]
            print('Nit: {}'.format(nit) +
                  ' -- Relative Dual Gap: {}'.format(gaps[0]) +
                  ' -- Relative Dual Gap: {}'.format(gaps[nit]) +
                  ' -- Objective: {}'.format(prec) +
                  ' -- Primal: {} --'.format(primals[nit]) +
                  ' --Dual: {}\n'.format(duals[nit]))
        else:
            primals[nit] = primals[nit - 1]
            duals[nit] = duals[nit - 1]
            gaps[nit] = gaps[nit - 1]

        nit += 1

        if nit == maxit:
            print('BEWARE, BAD CONVERGENCE, CHECK PARAMETERS !')

    return img_den, gaps, primals, duals, lambda_, q


def PhiStar(lambda_, p, C, alpha):
    """
    This function computes Phi^*(lambda)
    where:
    Phi(x)=||x||_1 if p=1
    Phi(x)=1/2||x||_2^2 if p=2
    Phi(x)=0 if p=Infty and ||x||_inf<=1 inf otherwise
    """
    if p == 1:
        v = np.sum(np.maximum(0, C * (np.abs(lambda_) - alpha)))
    elif p == 2:
        val = np.abs(lambda_ / alpha)
        v = alpha * (np.sum(np.minimum(val, C) * val -
                            0.5 * np.minimum(val, C) ** 2))
    else:
        v = np.minimum(C, alpha) * np.sum(np.abs(lambda_))
    return v


def Phi(x, p, alpha):
    if p == 1:
        n = alpha * np.sum(np.abs(x))
    elif p == 2:
        n = alpha / 2 * np.sum(x ** 2)
    else:
        if np.max(np.abs(x)) > alpha:
            n = np.inf
        else:
            n = 0
    return n


def Prox_Phi(x, p, M, tau, alpha, C):
    """
    This function solves :
    argmin_{|y|<=C} tau alpha ||y||_p + 1/2 ||M(y-x)||_2^2
    """
    if tau == 0:
        y = x / (np.maximum(1, np.abs(x) / C))
    else:
        if p == 1:
            tau = alpha * tau;
            y = np.maximum(np.abs(M * x) - tau, 0)
            y *= np.sign(x) / (np.maximum(np.abs(y) / C, 1))
        elif p == 2:
            tau = tau * alpha
            y = M * x / (tau + M)
            y /= (np.maximum(1, np.abs(y) / C))
        else:
            delta = np.minimum(alpha, C)
            y = x / (np.maximum(1, np.abs(x) / delta))
    return y


def drond1(im):
    res = np.zeros_like(im)
    shape = im.shape
    res[:shape[0] - 1, :] = im[1:shape[0], :] - im[:shape[0] - 1, :]
    return res


def drond1T(im):
    res = np.zeros_like(im)
    shape = im.shape
    res[1:shape[0] - 1, :] = im[:shape[0] - 2, :] - im[1:shape[0] - 1, :]
    res[0, :] = -im[0, :]
    res[shape[0] - 1, :] = im[shape[0] - 2, :]
    return res


def drond2(im):
    res = np.zeros_like(im)
    shape = im.shape
    res[:, :shape[1] - 1] = im[:, 1:shape[1]] - im[:, :shape[1] - 1]
    return res


def drond2T(im):
    res = np.zeros_like(im)
    shape = im.shape
    res[:, 1:shape[1] - 1] = im[:, :shape[1] - 2] - im[:, 1:shape[1] - 1]
    res[:, 0] = -im[:, 0]
    res[:, shape[0] - 1] = im[:, shape[0] - 2]
    return res
