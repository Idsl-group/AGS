import torch
import math
import torch.nn.functional as F
import numpy as np

def attr_batch_kernelSharp(image, kernel_size,h,v=1):
    empty_image = torch.zeros_like(image)
    pad=math.floor(kernel_size/2)
    padimg= F.pad(image,(pad,pad,pad,pad))
    h= F.pad(h,(pad,pad,pad,pad))
    h = 0.8*h + 0.169
    #h=1/(h*5+1)# got 0.926
    empty_padimg = torch.zeros_like(padimg)
    for i in range(pad, padimg.shape[-2] - pad):
        for j in range(pad, padimg.shape[-1] - pad):
            roi = padimg[:, :, i-pad:i+pad+1, j-pad:j+pad+1]
            #roi = padimg[:, :, i-pad:i+1, j-pad:j+1]
            empty_padimg[:, :, i:i+1, j:j+1] = attr_batch_smallSharp(roi,pad,h[:,:,i-pad:i-pad+1, j-pad:j-pad+1],v) #sharpening middle value
    return empty_padimg[:, :, pad:-pad, pad:-pad]

def attr_batch_smallSharp(X,pad,h=1,v=1): #sharpens single point in a matrix
    h=h/(np.sqrt(2)**v)
    Xsize = X.size()
    X = X.reshape(len(X), 1, -1)
    h = h.reshape(len(h), 1, -1)
    num=X.shape[1]
    for iter in range(0,v):
        a = X.repeat(1, X.shape[-1], 1).cuda()
        b = torch.ones(len(X), X.shape[-1], X.shape[-1]).cuda() #.repeat(len(X), 1, 1)
        b = b*X
        c = b - X.reshape(len(X), -1, 1)
        wts = norm_pdf(c, std=h)
        Ans = (a*wts).sum(2)/wts.sum(2)#.sum(0).sum(1)
        X = Ans
    XSharp=torch.reshape(X,Xsize)
    out = XSharp[:,:,pad:pad+1,pad:pad+1]
    return out

def norm_pdf(X,mean=0,std=1):
    try:
      std = std.reshape(len(X), 1, 1)
    except AttributeError:
      pass
    return torch.exp((-((X-mean)/std)**2)/2)*1/(std*np.sqrt(2*np.pi))

