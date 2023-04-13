from tqdm import tqdm
import torch
import torch.nn.functional as F


def compute_betas_least_square(
        X_torch,
        p_torch,
        M_torch,
        alpha,
        image_size,
        positions,
        rank="cpu",
        half_precision=False,
        verbose=True
):

    if half_precision:
        X_torch = X_torch.float()
        p_torch = p_torch.float()
        M_torch = M_torch.float()

    # send everything to the current GPU / device
    X_torch = X_torch.to(rank)
    p_torch = p_torch.to(rank)
    M_torch = M_torch.to(rank)

    eye = torch.eye(image_size ** 2,
                    image_size ** 2,
                    device=rank)

    # convolve the data
    X_conv = F.conv2d(X_torch, p_torch, padding="same")
    X_conv = X_conv.view(X_torch.shape[0], -1)

    X_conv_square = X_conv.T @ X_conv

    # Compute all betas in a loop over all positions
    betas = []

    tmp_iter = positions
    if verbose:
        tmp_iter = tqdm(positions)

    for x, y in tmp_iter:
        tmp_idx = x * image_size + y

        # get the current mask
        m_torch = M_torch[tmp_idx].flatten()
        Y_torch = X_torch.view(X_torch.shape[0], -1)[:, tmp_idx]

        # set up the least square problem
        lhs = ((X_conv_square * m_torch).T * m_torch).T + eye * alpha
        rhs = (X_conv * m_torch).T @ Y_torch

        # compute beta
        beta = torch.linalg.lstsq(lhs, rhs.view(-1, 1))

        # Maks the values of beta
        beta_cut = (beta * m_torch)
        betas.append(beta_cut)

    # Convolve the betas
    betas_conv = torch.stack(betas).reshape(
        len(betas), 1, image_size, image_size)

    beta_conv = F.conv2d(betas_conv, p_torch, padding="same")

    # bring everything back to the CPU
    beta_conv = beta_conv.squeeze().cpu()

    return beta_conv


def compute_betas_svd(
        X_torch,
        p_torch,
        M_torch,
        alphas,
        image_size,
        positions,
        rank="cpu",
        approx_svd=-1,
        half_precision=False,
        verbose=True):

    if half_precision:
        X_torch = X_torch.float()
        p_torch = p_torch.float()
        M_torch = M_torch.float()

    # send everything to the current GPU / device
    X_torch = X_torch.to(rank)
    p_torch = p_torch.to(rank)
    M_torch = M_torch.to(rank)

    # convolve the data
    X_conv = F.conv2d(X_torch, p_torch, padding="same")
    X_conv = X_conv.view(X_torch.shape[0], -1)

    # Compute all betas in a loop over all positions
    betas = []

    tmp_iter = positions
    iter_alpha = alphas
    if verbose:
        tmp_iter = tqdm(positions)
        iter_alpha = tqdm(alphas)

    for x, y in tmp_iter:
        tmp_idx = x * image_size + y

        # get the current mask
        m_torch = M_torch[tmp_idx].flatten()
        Y_torch = X_torch.view(X_torch.shape[0], -1)[:, tmp_idx]

        # mask the data
        X_conv_cut = X_conv * m_torch

        # Compute the SVD
        if approx_svd != -1:
            svd_out = torch.svd_lowrank(
                X_conv_cut,
                niter=1,
                q=approx_svd)
        else:
            svd_out = torch.linalg.svd(
                X_conv_cut,
                full_matrices=False,
                driver="gesvd")

        U_torch = svd_out.U
        D_torch = svd_out.S
        V_torch = svd_out.Vh.T

        # compute the betas
        local_betas = []
        rhs = torch.diag(D_torch) @ U_torch.T @ Y_torch
        for tmp_alpha in iter_alpha:
            eye = torch.ones_like(D_torch, device=rank) * tmp_alpha
            # 1D vector
            inv_eye = 1 / (D_torch ** 2 + eye)

            # compute beta
            # TODO check this
            # beta = (V_torch.T * inv_eye).T @ rhs
            beta = V_torch * inv_eye @ rhs

            # cut the beta
            beta_cut = (beta * m_torch)

            local_betas.append(beta_cut)

        # stack and convolve the current betas
        tmp_betas = torch.stack(local_betas).reshape(
            len(local_betas), 1, image_size, image_size)

        tmp_betas_conv = F.conv2d(
            tmp_betas, p_torch, padding="same")

        betas.append(tmp_betas_conv.squeeze().cpu())

    # Stack all results and return them
    betas_final = torch.stack(betas)

    return betas_final
