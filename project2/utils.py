def TVLoss(model, layer_mask):
    conv2D_idxs = [0, 3, 6, 8, 10]

    features = list(model.features.children())
    tv = 0

    for layer in layer_mask:
        weights = list(features[conv2D_idxs[layer]].parameters())[0]
        size = weights.size()

        for kernel in range(size[0]):
            for channel in range(size[1]):
                y = weights[kernel][channel]

                for i in range(size[2] - 1):
                    for j in range(size[3] - 1):
                        tv += torch.sqrt((y[i + 1][j] - y[i][j])**2 + (y[i][j + 1] - y[i][j])**2)
    
    return tv
