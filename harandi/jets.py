"""Exact value/Jacobian/Hessian propagation for ParallelMLP tanh networks.

Derivatives are with respect to the supplied input's first spatial_dim
coordinates. For normalized z=2*x-1, multiply Jacobians by 2 and Hessians by 4.
All operations remain differentiable with respect to network parameters. Each
linear layer uses one batched matrix multiplication for all derivative channels.
"""
import torch


def forward_jet(network, z, spatial_dim=2, second_order=False):
    """Return (value, jacobian[, hessian]) without spatial autograd calls.

    z: [heads, points, input_dim]
    value: [heads, points]
    jacobian: [heads, points, spatial_dim]
    hessian, if requested: [heads, points, spatial_dim, spatial_dim]

    The network must expose ParallelMLP's weights/biases: tanh hidden layers,
    independent heads, and one linear scalar output per head. Additional input
    dimensions, e.g. global material ratios, remain constant for these spatial
    derivatives but still participate normally in parameter differentiation.
    """
    if z.ndim != 3 or not 1 <= spatial_dim <= z.shape[-1]:
        raise ValueError('z must be [heads,points,inputs] with valid spatial_dim')
    heads, points, inputs = z.shape
    if points < 1 or network.weights[0].shape[:2] != (heads, inputs):
        raise ValueError('input shape does not match network heads/input dimension')
    if network.weights[-1].shape[-1] != 1:
        raise ValueError('forward_jet requires one scalar output per head')
    value = z
    jac = torch.eye(spatial_dim, inputs, device=z.device, dtype=z.dtype)
    jac = jac[None, :, None, :].expand(heads, -1, points, -1)
    hess = (z.new_zeros((heads, spatial_dim, spatial_dim, points, inputs))
            if second_order else None)
    channels = 1+spatial_dim+(spatial_dim*spatial_dim if second_order else 0)
    for index, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        parts = (value[:, None], jac)
        if second_order:
            parts += (hess.reshape(heads, spatial_dim*spatial_dim, points, value.shape[-1]),)
        packed = torch.cat(parts, dim=1).reshape(heads, channels*points, value.shape[-1])
        linear = torch.bmm(packed, weight).reshape(heads, channels, points, weight.shape[-1])
        value = linear[:, 0]+bias
        jac = linear[:, 1:1+spatial_dim]
        if second_order:
            hess = linear[:, 1+spatial_dim:].reshape(heads, spatial_dim, spatial_dim, points, -1)
        if index < len(network.weights)-1:
            value = torch.tanh(value)
            slope = 1-value.square()
            if second_order:
                hess = (slope[:, None, None]*hess
                        - (2*value*slope)[:, None, None]*jac[:, :, None]*jac[:, None, :])
            jac = slope[:, None]*jac
    result = (value[..., 0], jac[..., 0].permute(0, 2, 1))
    if second_order:
        result += (hess[..., 0].permute(0, 3, 1, 2),)
    return result
