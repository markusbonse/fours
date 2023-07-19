from tqdm import tqdm
import torch
import torch.nn.functional as F


def compute_betas_least_square(
        X_torch,
        M_torch,
        lambda_reg,
        positions,
        p_torch=None,
        verbose=True,
        device="cpu"
):
    """
    X_torch: (num_images, images_size, images_size) (normalized!)
    M_torch: (images_size**2, images_size**2)
    p_torch: (x, y, 1, 1)
    """

    X_torch = X_torch.unsqueeze(1)
    image_size = X_torch.shape[-1]

    # convolve the data
    if p_torch is not None:
        X_conv = F.conv2d(X_torch, p_torch, padding="same")
    else:
        X_conv = X_torch

    X_conv = X_conv.view(X_torch.shape[0], -1)

    # move the convolved data to the GPU
    X_conv = X_conv.to(device)
    X_conv_square = X_conv.T @ X_conv

    # Compute all betas in a loop over all positions
    betas = []

    tmp_iter = positions
    if verbose:
        tmp_iter = tqdm(positions)

    for x, y in tmp_iter:
        tmp_idx = x * image_size + y

        # get the current mask
        m_torch = M_torch[tmp_idx].flatten().to(device)
        Y_torch = X_torch.view(X_torch.shape[0], -1)[:, tmp_idx].to(device)

        # set up the least square problem
        lhs = ((X_conv_square * m_torch).T * m_torch).T + \
              torch.eye(
                  image_size ** 2,
                  image_size ** 2,
                  device=X_conv_square.device) * lambda_reg

        rhs = (X_conv * m_torch).T @ Y_torch

        # compute beta
        beta = torch.linalg.lstsq(lhs, rhs.view(-1, 1))

        betas.append(beta.solution.squeeze())

    # We move the beta matrix to the device of the noise model (usually cpu)
    betas_final = torch.stack(betas).reshape(
        len(betas), image_size**2).to(M_torch.device)

    return betas_final


def compute_betas_svd(
        X_torch,
        M_torch,
        lambda_regs,
        positions,
        p_torch=None,
        approx_svd=-1,
        verbose=True,
        device="cpu"):

    image_size = X_torch.shape[-1]
    X_torch = X_torch.unsqueeze(1)

    # convolve the data on the GPU
    X_torch = X_torch.to(device)
    if p_torch is not None:
        X_conv = F.conv2d(X_torch.to(device),
                          p_torch.to(device), padding="same")
    else:
        X_conv = X_torch.to(device)
    X_torch = X_torch.cpu()

    X_conv = X_conv.view(X_torch.shape[0], -1)

    # Compute all betas in a loop over all positions
    betas = []

    tmp_iter = positions
    if verbose:
        tmp_iter = tqdm(positions)

    for x, y in tmp_iter:
        tmp_idx = x * image_size + y

        # get the current mask
        m_torch = M_torch[tmp_idx].flatten().to(device)
        Y_torch = X_torch.view(X_torch.shape[0], -1)[:, tmp_idx].to(device)

        # mask the data
        X_conv_cut = X_conv * m_torch

        # Compute the SVD
        if approx_svd != -1:
            svd_out = torch.svd_lowrank(
                X_conv_cut,
                niter=1,
                q=approx_svd)
            U_torch = svd_out[0]
            D_torch = svd_out[1]
            V_torch = svd_out[2]
        else:
            svd_out = torch.linalg.svd(
                X_conv_cut,
                full_matrices=False,
                driver="gesvd")

            U_torch = svd_out.U
            D_torch = svd_out.S
            V_torch = svd_out.Vh.T

        # clean up memory
        del X_conv_cut

        # compute the betas
        local_betas = []
        rhs = torch.diag(D_torch) @ U_torch.T @ Y_torch
        for tmp_lambda_reg in lambda_regs:
            eye = torch.ones_like(
                D_torch,
                device=D_torch.device) * tmp_lambda_reg

            # 1D vector
            inv_eye = 1 / (D_torch ** 2 + eye)

            # compute beta
            beta = V_torch * inv_eye @ rhs

            # cut the beta
            beta_cut = (beta * m_torch)

            local_betas.append(beta_cut)

        # stack and convolve the current betas
        tmp_betas = torch.stack(local_betas).reshape(
            len(local_betas), 1, image_size, image_size)

        if p_torch is not None:
            tmp_betas_conv = F.conv2d(
                tmp_betas, p_torch.to(device), padding="same")
        else:
            tmp_betas_conv = tmp_betas

        betas.append(tmp_betas_conv.squeeze().to(M_torch.device))

    # Stack all results and return them
    betas_final = torch.stack(betas)

    return betas_final
