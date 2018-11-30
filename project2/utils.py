import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as ndimage
import cv2
from torch.nn import functional as F
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

# Acknowledgment: taken from
# https://zhuanlan.zhihu.com/p/31421408?fbclid=IwAR2diRqcW8pPm_BxrXgkSK_E29YOeohzP9IOvKELcuTS1K7gLM4rvlv3Bos
def compute_saliency_map(X, y, model, device):
    model.eval()
    X_var = Variable(X, requires_grad=True).to(device)
    y_var = Variable(y).to(device)
    saliency = None

    scores = model(X_var)
    scores = scores.gather(1, y_var.view(-1, 1)).squeeze()
#    scores,_ = torch.max(scores,1)
    scores.backward(torch.FloatTensor([1.0] * X_var.shape[0]).to(device))
    
    saliency = X_var.grad.data
    saliency = saliency.abs()
    saliency, i = torch.max(saliency,dim=1)
    saliency = saliency.squeeze()
    return saliency

def save_saliency_map(saliency_map, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(saliency_map, f)

def show_all_saliency_maps(saliency_maps, image):
    num_models = len(saliency_maps)
    num_classes = saliency_maps[0].shape[0]
    
    for i in range(num_classes):            # iterating over classes
        for j in range(num_models):         # iterating over models
            x = np.moveaxis(image[i,:,:,:].cpu().numpy(),0,2)
            saliency = saliency_maps[j][i,:,:].cpu().numpy()
            saliency = ndimage.gaussian_filter(saliency, sigma=1)  
            plt.subplot(num_classes, num_models*2, 1 + (i * num_models*2) + j*2)
            plt.imshow(saliency)
            plt.subplot(num_classes, num_models*2, 1 + (i * num_models*2) + 2*j+1)
            plt.imshow(x)
            plt.gcf().set_size_inches(24,24)
    plt.savefig('yeet.png')
       
def show_all_CAM(CAM_maps, image, filename):
    num_models = len(CAM_maps)
    num_classes = len(CAM_maps[0])
    print(num_classes)
    _,_,height,width = image.shape
    vstack = []
    for i in range(num_classes):    
        hstack = []
        for j in range(num_models):         # iterating over models
            x = cv2.resize((np.moveaxis(image[0,:,:,:].cpu().numpy(),0,2) * 255).astype('uint8'),(64,64))
            saliency = CAM_maps[j][i][:,:]
            heatmap = cv2.applyColorMap(cv2.resize(saliency,(64,64)), cv2.COLORMAP_JET)
            result = heatmap * 0.5 + x * 0.7
            hstack.append(result.copy())
        vstack.append(np.hstack(hstack))
    final = np.vstack(vstack)
    cv2.imwrite(filename,final)


#https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        print(feature_conv.shape)
        cam = weight_softmax[idx].dot(feature_conv[0,:,:,:].reshape((nc, h*w)))
        print(cam.shape)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def CAM(X, y, model, features_blobs, weight_softmax, device):
    model.eval()
    X_var = Variable(X, requires_grad=True).to(device)
    y_var = Variable(y).to(device)

    scores = model(X_var)
    h_scores = F.softmax(scores, dim=1).data.squeeze()
    probs, idx = h_scores.sort(0,True)
#    print(probs, idx)
#    print(h_scores.shape)
#    print(probs.shape)
#    print(idx.shape)    
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    
    return returnCAM(features_blobs[0], weight_softmax, [idx[0]]) 

