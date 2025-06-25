import torch
import torch.nn.functional as F

def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]


def matting_loss(preds, true_pha, weights=None):
    """
    Compute matting loss with optional deep supervision.

    Args:
        preds (Tensor or list of Tensors): Predicted alpha matte(s), each of shape (B, 1, H, W).
        true_pha (Tensor): Ground truth alpha matte, shape (B, 1, H, W).
        weights (list of float, optional): Weights for each prediction if deep supervision is enabled.

    Returns:
        dict: {
            'pha_l1': total L1 loss,
            'pha_laplacian': total Laplacian loss,
            'total': total combined loss
        }
    """

    # Ensure preds is a list
    if not isinstance(preds, list):
        preds = [preds]

    # Default weights: full weight for final output, smaller for intermediates
    if weights is None:
        weights = [1.0] + [0.5] * (len(preds) - 1)

    # Initialize losses
    total_loss = 0.0
    l1_total = 0.0
    lap_total = 0.0

    for pred, weight in zip(preds, weights):
        l1 = F.l1_loss(pred, true_pha)
        lap = laplacian_loss(pred, true_pha)
        total = l1 + lap

        l1_total += l1 * weight
        lap_total += lap * weight
        total_loss += total * weight

    return {
        'pha_l1': l1_total,
        'pha_laplacian': lap_total,
        'total': total_loss
    }
