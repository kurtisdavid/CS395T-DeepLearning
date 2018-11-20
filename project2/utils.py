import torch

def TVLoss(model, layer_mask=None):
    conv2D_idxs = [0, 3, 6, 8, 10]
    if layer_mask is None:
        layer_mask = [i for i in range(len(conv2D_idxs))]

    features = list(model.features.children())
    tv = 0

    pixels = 0
    for layer in layer_mask:
        weights = list(features[conv2D_idxs[layer]].parameters())[0]
        size = weights.size()

        pixels += size[0] * size[1] * (size[2] - 1) * (size[3] - 1)

        for kernel in range(size[0]):
            for channel in range(size[1]):
                y = weights[kernel][channel]

                for i in range(size[2] - 1):
                    for j in range(size[3] - 1):
                        tv += torch.sqrt((y[i + 1][j] - y[i][j])**2 + (y[i][j + 1] - y[i][j])**2)

    return tv / pixels

# big yeet
def check_grad(model):
    conv2D_idxs = [0, 3, 6, 8, 10]
    features = list(model.features.children())

    for layer in conv2D_idxs:
        weights = list(features[layer].parameters())[0]
        x = torch.zeros_like(weights)
        y = torch.zeros_like(weights)
        x[:, :, :-1, :-1] = (weights[:, :, 1:, :-1] - weights[:, :, :-1, :-1]) ** 2
        y[:, :, :-1, :-1] = (weights[:, :, :-1, 1:] - weights[:, :, :-1, :-1]) ** 2
        yeet = x + y
        huh = torch.sqrt(yeet)
        ok = torch.sum(huh)
        if torch.sum(torch.isnan(weights.grad)).item()>0:
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    for k in range(weights.shape[2]):
                        for l in range(weights.shape[3]):
                            if weights.grad[i,j,k,l]!=weights.grad[i,j,k,l]:
                                print(weights[i,j,:,:])
                                print(weights.grad[i,j,:,:])
                                print(x[i,j,:,:])
                                print(y[i,j,:,:])
                                print(yeet[i,j,:,:])
                                print(huh[i,j,:,:])
                                print(ok[i,j,:,:])
                                exit()

def TVLossMat(model, print_=False, layer_mask=None):
    conv2D_idxs = [0, 3, 6, 8, 10]
    if layer_mask is None:
        layer_mask = [i for i in range(len(conv2D_idxs))]
    features = list(model.features.children())
    tv = 0
    tvs = []
    for layer in layer_mask:
        weights = list(features[conv2D_idxs[layer]].parameters())[0]
        size = weights.size()
#         print("size: ", size)

        pixels = size[0] * size[1] * (size[2] - 1) * (size[3] - 1)
        curr = TVMat(weights)/pixels
        tvs.append(curr.item())
        tv += curr
        if print_:
            pass
            # print("sum", yeet)
            # print("sqrt",huh)
            # print("ok",ok)
            # print("tv", tv)
            # print("pixels", pixels)
    # if print_:
    #     print("final_tv", tv)
    #     print("final_pixels", pixels)
    return tv/len(layer_mask), tvs


def TVMat(weights):
    x = torch.zeros_like(weights)
    y = torch.zeros_like(weights)
    x[:, :, :-1, :-1] = (weights[:, :, 1:, :-1] - weights[:, :, :-1, :-1]) ** 2
    y[:, :, :-1, :-1] = (weights[:, :, :-1, 1:] - weights[:, :, :-1, :-1]) ** 2
    return torch.sum(torch.sqrt(x+y+1e-6))

def TVMat3D(weights):
    x = torch.zeros_like(weights)
    y = torch.zeros_like(weights)
    z = torch.zeros_like(weights)

    x[:, :-1, :-1, :-1] = (weights[:, :-1, 1:, :-1] - weights[:, :-1, :-1, :-1]) ** 2
    y[:, :-1, :-1, :-1] = (weights[:, :-1, :-1, 1:] - weights[:, :-1, :-1, :-1]) ** 2
    z[:, :-1, :-1, :-1] = (weights[:, 1:, :-1, :-1] - weights[:, :-1, :-1, :-1]) ** 2

    # regularize the last channel
    z_x = (weights[:, -1, 1:, :-1] - weights[:, -1, :-1, :-1]) ** 2
    z_y = (weights[:, -1, :-1, 1:] - weights[:, -1, :-1, :-1]) ** 2

    return torch.sum(torch.sqrt(x + y + z + 1e-6)) + torch.sum(torch.sqrt(z_x + z_y + 1e-6))

def TVMat4D(weights):
    x = torch.zeros_like(weights)
    y = torch.zeros_like(weights)
    z = torch.zeros_like(weights)
    a = torch.zeros_like(weights)
    a_x = torch.zeros_like(weights)
    a_y = torch.zeros_like(weights)
    a_z = torch.zeros_like(weights)

    x[:-1, :-1, :-1, :-1] = (weights[:-1, :-1, 1:, :-1] - weights[:-1, :-1, :-1, :-1]) ** 2
    y[:-1, :-1, :-1, :-1] = (weights[:-1, :-1, :-1, 1:] - weights[:-1, :-1, :-1, :-1]) ** 2
    z[:-1, :-1, :-1, :-1] = (weights[:-1, 1:, :-1, :-1] - weights[:-1, :-1, :-1, :-1]) ** 2
    a[:-1, :-1, :-1, :-1] = (weights[1:, :-1, :-1, :-1] - weights[:-1, :-1, :-1, :-1]) ** 2

    # regularize the last cube
    a_x[-1, :-1, :-1, :-1] = (weights[-1, :-1, 1:, :-1] - weights[-1, :-1, :-1, :-1]) ** 2
    a_y[-1, :-1, :-1, :-1] = (weights[-1, :-1, :-1, 1:] - weights[-1, :-1, :-1, :-1]) ** 2
    a_z[-1, :-1, :-1, :-1] = (weights[-1, 1:, :-1, :-1] - weights[-1, :-1, :-1, :-1]) ** 2

    # regularize the last channel
    z_x = (weights[:, -1, 1:, :-1] - weights[:, -1, :-1, :-1]) ** 2
    z_y = (weights[:, -1, :-1, 1:] - weights[:, -1, :-1, :-1]) ** 2

    return torch.sum(torch.sqrt(x + y + z + a + 1e-6)) \
        + torch.sum(torch.sqrt(a_x + a_y + a_z + 1e-6)) \
        + torch.sum(torch.sqrt(z_x + z_y + 1e-6))

def TVLossResNet(model, layer_mask=None, TVLoss=TVMat):

    blocks = [model.conv1, model.layer1, model.layer2, model.layer3]
    if layer_mask is None:
        layer_mask = [i for i in range(len(blocks))]

    # get tv values for convolutions in resnet block
    def TVBlock(block):
        block = list(block.children())
        tv = 0
        weights = list(block[0].parameters())[0]
        size = weights.size()
        pixels = size[0] * size[1] * (size[2] - 1) * (size[3] - 1)
        tv += TVLoss(weights)/pixels

        weights = list(block[3].parameters())[0]
        size = weights.size()
        pixels = size[0] * size[1] * (size[2] - 1) * (size[3] - 1)
        tv += TVLoss(weights)/pixels

        return tv/2

    tv = 0
    tvs = []
    for layer in layer_mask:
        if layer == 0:
            weights = list(blocks[layer].parameters())[0]
            size = weights.size()
            pixels = size[0] * size[1] * (size[2] - 1) * (size[3] - 1)
            curr = TVLoss(weights)/pixels
            tv += curr
            tvs.append(curr.item())
        else:
            basic_blocks = list(blocks[layer].children())
            total = 0
            for bl in basic_blocks:
                curr = TVBlock(bl)
                tv += curr
                total += curr
            tvs.append(total/len(basic_blocks))
    return tv/len(layer_mask), tvs
