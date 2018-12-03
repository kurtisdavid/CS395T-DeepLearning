from scipy.fftpack import dct, idct
from scipy.signal import resample
from scipy.interpolate import griddata
import numpy as np
import torch

def compress_weights(weights, ratio, scheme):
    old_weights = weights.detach().cpu().numpy()
    size = weights.size()
    if scheme == 'dct':
        scheme = compress_weights_dct
    elif scheme == 'resample':
        scheme = compress_weights_resample
    elif scheme == 'resize':
        scheme = compress_weights_resize
    new_weights = torch.FloatTensor(scheme(weights, ratio, size))
    new_size = [int(np.ceil(size[i] * ratio)) for i in range(4)]
    for a in range(size[0]):
        for b in range(size[1]):
            for c in range(size[2]):
                for d in range(size[3]):
                    weights[a,b,c,d] = new_weights[a,b,c,d]
    norm = np.sum(np.abs(weights.detach().cpu().numpy() - old_weights))
    return norm, new_size[0] * new_size[1] * new_size[2] * new_size[3]

def compress_weights_dct(weights, ratio, size):
    new_size = [int(np.ceil(size[i] * ratio)) for i in range(4)]
    return idct(idct(idct(idct(dct(dct(dct(dct(
        weights.detach().cpu().numpy(),
        axis=0, norm='ortho')[:new_size[0], :, :, :],
        axis=1, norm='ortho')[:, :new_size[1], :, :],
        axis=2, norm='ortho')[:, :, :new_size[2], :],
        axis=3, norm='ortho')[:, :, :, :new_size[3]],
        n=size[3], axis=3, norm='ortho'),
        n=size[2], axis=2, norm='ortho'),
        n=size[1], axis=1, norm='ortho'),
        n=size[0], axis=0, norm='ortho')

def compress_weights_resample(weights, ratio, size):
    new_size = [int(np.ceil(size[i] * ratio)) for i in range(4)]
    return resample(resample(resample(resample(resample(resample(resample(resample(
        weights.detach().cpu().numpy(), new_size[0], axis=0),
        new_size[1], axis=1),
        new_size[2], axis=2),
        new_size[3], axis=3),
        size[3], axis=3),
        size[2], axis=2),
        size[1], axis=1),
        size[0], axis=0)

def compress_weights_resize(weights, ratio, size):
    new_size = [int(np.ceil(size[i] * ratio)) for i in range(4)]
    x = np.mgrid[:size[0], :size[1], :size[2], :size[3]].reshape(4,-1).T
    a, b, c, d = np.mgrid[:new_size[0], :new_size[1], :new_size[2], :new_size[3]]
    arr = griddata(x, weights.detach().cpu().numpy().reshape(-1), (a, b, c, d), method='nearest')
    x = np.mgrid[:new_size[0], :new_size[1], :new_size[2], :new_size[3]].reshape(4,-1).T
    a, b, c, d = np.mgrid[:size[0], :size[1], :size[2], :size[3]]
    return griddata(x, arr.reshape(-1), (a, b, c, d), method='nearest')

def compress(model, ratio, layer_mask, device, scheme='dct'):
    blocks = [model.conv1, model.layer1, model.layer2, model.layer3]
    if layer_mask is None:
        layer_mask = [i for i in range(len(blocks))]

    old_params_count = 0
    new_params_count = 0
    l2_diff = 0
    for layer in layer_mask:
        if layer == 0:
            weights = list(blocks[layer].parameters())[0]
            size = weights.size()
            old_params_count += size[0] * size[1] * size[2] * size[3]
            norm, npc = compress_weights(weights, ratio, scheme)
            l2_diff += norm
            new_params_count += npc
        else:
            basic_blocks = list(blocks[layer].children())
            for block in basic_blocks:
                block = list(block.children())
                weights = list(block[0].parameters())[0]
                size = weights.size()
                old_params_count += size[0] * size[1] * size[2] * size[3]
                norm, npc = compress_weights(weights, ratio, scheme)
                l2_diff += norm
                new_params_count += npc

                weights = list(block[3].parameters())[0]
                size = weights.size()
                old_params_count += size[0] * size[1] * size[2] * size[3]
                norm, npc = compress_weights(weights, ratio, scheme)
                l2_diff += norm
                new_params_count += npc
    return l2_diff / old_params_count, new_params_count / old_params_count
