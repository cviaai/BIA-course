import torch
import numpy as np
from scipy.fftpack import dct, dctn
from functools import reduce
import math
import torch.nn as nn
import random
import torch.nn.functional as F
from networks import UNet


def gen_dct2_kernel(support,dtype = 'f', GPU = False, nargout = 1):
    r"""If two output arguments are returned then:
    h1 has dimensions [Px,1,Px,1] and h2 has dimensions [Py,1,1,Py] where
    Px = support[0] and  Py = support[1].
    
    If a single output argument is returned then:
    h has dimensions [Px*Py 1 Px Py].
    
    support : Either an integer or a tuple
    
    Usage example :
    
    x = torch.randn(1,1,16,16).double()
    h = utils.gen_dct2_kernel(8,'d',nargout = 1)
    Dx = torch.conv2d(x,h,stride = (8,8)) % it computes the 2D DCT of each 
    % non-overlapping block of size 8x8
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = torch.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())
    
    Usage example 2:
    
    x = torch.randn(1,1,16,16).double()
    h1,h2 = utils.gen_dct2_kernel(8,'d',nargout = 2)
    Dx = torch.conv2d(x,h1,stride = (8,1)) % it computes the 1D DCT of each 
    % non-overlapping block of size 8x1
    Dx = Dx.view(8,1,2,1)
    Dx = torch.conv2d(Dx,h2,stride = (1,8)) % it computes the 1D DCT of each
    % non-overlapping block of size 1x8
    Dx = Dx.view(1,64,2,2)
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = torch.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())    
    
    """
    assert(nargout == 1 or nargout ==2), "One or two output arguments "\
    "are expected."
    
    
    if isinstance(support,int):
        support = (support,support)
    
    if len(support) < 2:
        support = support * 2
        
    if nargout == 2:
        D = dctmtx(support[0],dtype)
        h1 = D.view(support[0],1,support[0],1)
        if support[1] != support[0]:
            D = dctmtx(support[1],dtype)
        h2 = D.view(support[1],1,1,support[1])
        
        if torch.cuda.is_available and GPU:
            h1 = h1.cuda()
            h2 = h2.cuda()
                   
        return h1,h2
    else :
        h = np.zeros((reduce(lambda x, y: x*y, support[0:2]),1,support[0],support[1]),dtype)
        dirac = np.zeros(support[0:2],dtype)
        for k in np.arange(support[0]):
            for l in np.arange(support[1]):
                dirac[k,l] = 1;
                h[:,0,k,l] = dctn(dirac,norm = 'ortho').flatten()
                dirac[k,l] = 0
        
        h = torch.from_numpy(h)
        if torch.cuda.is_available and GPU:
            h = h.cuda()
        
        return h

def gen_dct3_kernel(support,dtype = 'f', GPU = False, nargout = 1):
    r"""If three output arguments are returned then:
    h1 has dimensions [Pz,Pz,1,1], h2 has dimensions [Px,1,Px,1] and 
    h3 has dimensions [Py,1,1,Py] where Pz = support[0], Px = support[1] and 
    Py = support[2].
    
    If two output arguments are returned then:
    h1 has dimensions [Pz,Pz,1,1] and h2 has dimensions [Px*Py,1,Px,Py].
    
    If a single output argument is returned then:
    h has dimensions [Px*Py*Pz Pz Px Py].
    
    support : Either an integer or a tuple 
    
    Usage example :
    from scipy.fftpack import dctn
    x = torch.randn(1,3,16,16).double()
    h = utils.gen_dct3_kernel((3,8,8),'d',nargout = 1)
    Dx = torch.conv2d(x,h,stride = (8,8)) % it computes the 3D DCT of each 
    % non-overlapping block of size 3x8x8
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = torch.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())
    
    Usage example 2:
    
    x = torch.randn(1,3,16,16).double()
    h1,h2 = utils.gen_dct3_kernel((3,8,8),'d',nargout = 2)
    Dx = torch.conv2d(x,h1,stride = 1) % it computes the 1D DCT along the 3rd  
    % dimension.
    Dx = torch.conv2d(Dx.view(3,1,16,16),h2,stride = (8,8)) % it computes the 2D 
    % DCT along the spatial dimensions
    Dx = Dx.view(1,3*64,2,2)
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = torch.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())

    Usage example 3:
    
    x = torch.randn(1,3,16,16).double()
    h1,h2,h3 = utils.gen_dct3_kernel((3,8,8),'d',nargout = 3)
    Dx = torch.conv2d(x,h1,stride = 1) % it computes the 1D DCT along the 3rd 
    % dimension. 
    Dx = torch.conv2d(Dx.view(3,1,16,16),h2,stride = (8,1)) % it computes the 1D 
    % DCT along the first spatial dimension
    Dx = torch.conv2d(Dx.view(3*8,1,2,16),h3,stride = (1,8)) % it computes the 1D DCT along the channel
    % dimension (of size 3)
    Dx = Dx.view(1,24*8,2,2)
    
    for k in range(2):
        for l in range(2):
            s = x[:,:,k*8:(k+1)*8,l*8:(l+1)*8].numpy().squeeze()
            Ds = torch.from_numpy(dctn(s,norm = 'ortho').flatten())
            err = Ds - Dx[:,:,k,l]
            print(err.abs().sum())
    
    """
    assert(nargout == 1 or nargout ==2 or nargout == 3), "From one to three "\
    "output arguments are expected."
    
    
    if isinstance(support,int):
        support = (support,support,support)
    
    if len(support) < 2:
        support = support * 3
    
    if len(support) < 3:
        support = (1,)+support
    
    if nargout == 3:
        D = dctmtx(support[0],dtype)
        h1 = D.view(support[0],support[0],1,1)
        if support[1] != support[0]:
            D = dctmtx(support[1],dtype)
        h2 = D.view(support[1],1,support[1],1)
        if support[2] != support[1]:
            D = dctmtx(support[2],dtype)
        h3 = D.view(support[2],1,1,support[2])
        
        if torch.cuda.is_available and GPU:
            h1 = h1.cuda()
            h2 = h2.cuda()
            h3 = h3.cuda()
                   
        return h1,h2,h3            
    elif nargout == 2:
        D = dctmtx(support[0],dtype)
        h1 = D.view(support[0],support[0],1,1)

        h2 = np.zeros((reduce(lambda x, y: x*y, support[1:3]),1,support[1],support[2]),dtype)
        dirac = np.zeros(support[1:3],dtype)
        for k in np.arange(support[1]):
            for l in np.arange(support[2]):
                dirac[k,l] = 1;
                h2[:,0,k,l] = dctn(dirac,norm = 'ortho').flatten()
                dirac[k,l] = 0
        
        h2 = torch.from_numpy(h2)        
        
        if torch.cuda.is_available and GPU:
            h1 = h1.cuda()
            h2 = h2.cuda()
                   
        return h1,h2
    else :
        h = np.zeros((reduce(lambda x, y: x*y, support[0:3]),support[0],support[1],support[2]),dtype)
        dirac = np.zeros(support[0:3],dtype)
        for k in np.arange(support[0]):
            for l in np.arange(support[1]):
                for m in np.arange(support[2]):
                    dirac[k,l,m] = 1;
                    h[:,k,l,m] = dctn(dirac,norm = 'ortho').flatten()
                    dirac[k,l,m] = 0
        
        h = torch.from_numpy(h)
        if torch.cuda.is_available and GPU:
            h = h.cuda()
        
        return h
    
    
def periodicPad2D(input,pad = 0):
    r"""Pads circularly the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""
          
    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
             
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = torch.empty(sz).type_as(input)
    
    # Copy the original tensor to the central part
    out[...,pad[0]:out.size(-2)-pad[1], \
        pad[2]:out.size(-1)-pad[3]] = input
    
    # Pad Top
    if pad[0] != 0:
        out[...,0:pad[0],:] = out[...,out.size(-2)-pad[1]-pad[0]:out.size(-2)-pad[1],:]
    
    # Pad Bottom
    if pad[1] != 0:
        out[...,out.size(-2)-pad[1]::,:] = out[...,pad[0]:pad[0]+pad[1],:]
    
    # Pad Left
    if pad[2] != 0:
        out[...,:,0:pad[2]] = out[...,:,out.size(-1)-pad[3]-pad[2]:out.size(-1)-pad[3]]
    
    # Pad Right
    if pad[3] != 0:
        out[...,:,out.size(-1)-pad[3]::] = out[...,:,pad[2]:pad[2]+pad[3]]    
    
    if sflag:
        out.squeeze_()
        
    return out

def periodicPad_transpose2D(input,crop = 0):
    r"""Adjoint of the periodicPad2D operation which amounts to a special type
    of cropping. CROP specifies the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT].
    If crop is an integer then each direction is cropped by the same amount. In
    order to achieve a different amount of cropping in each direction of the 
    tensor, crop needs to be a tuple."""          
    
    # crop = [top,bottom,left,right]
    
    if isinstance(crop,int):
        assert(crop >= 0), """Crop must be either a non-negative integer 
        or a tuple."""
        crop = (crop,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)        
             
    assert(isinstance(crop,tuple) and len(crop) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0), \
            "Crop must be non-negative in each dimension."    
    
    assert (crop[0] + crop[1] <= sz[-2] and crop[2] + crop[3] <= sz[-1]), \
            "Crop does not have valid values."
    
    out = input.clone()
    
    # Top
    if crop[0] != 0:
        out[...,crop[0]:crop[0]+crop[1],:] += out[...,-crop[1]::,:]
    
    # Bottom 
    if crop[1] != 0:
        out[...,-crop[0]-crop[1]:-crop[1],:] += out[...,0:crop[0],:]
    
    # Left 
    if crop[2] != 0:
        out[...,crop[2]:crop[2]+crop[3]] += out[...,-crop[3]::]
    
    # Right
    if crop[3] != 0:
        out[...,-crop[2]-crop[3]:-crop[3]] += out[...,0:crop[2]]
    
    if crop[1] == 0:
        end_h = sz[-2]+1 
    else:
        end_h = sz[-2]-crop[1]
        
    if crop[3] == 0:
        end_w = sz[-1]+1
    else:
        end_w = sz[-1]-crop[3]
        
    out = out[...,crop[0]:end_h,crop[2]:end_w]
    
    if sflag:
        out.squeeze_()
        
    return out


def zeroPad2D(input,pad = 0):
    r"""Pads with zeros the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""

    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
        
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."    
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = torch.zeros(sz).type_as(input)
    out[...,pad[0]:sz[-2]-pad[1]:1,pad[2]:sz[-1]-pad[3]:1] = input
    
    if sflag:
        out.squeeze_()
    
    return out

def symmetricPad2D(input,pad = 0):
    r"""Pads symmetrically the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""
          
    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
             
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = torch.zeros(sz).type_as(input)
    
    # Copy the original tensor to the central part
    out[...,pad[0]:out.size(-2)-pad[1], \
        pad[2]:out.size(-1)-pad[3]] = input
    
    # Pad Top
    if pad[0] != 0:
        out[...,0:pad[0],:] = reverse(out[...,pad[0]:2*pad[0],:],-2)
    
    # Pad Bottom
    if pad[1] != 0:
        out[...,out.size(-2)-pad[1]::,:] = reverse(out[...,out.size(-2)
            -2*pad[1]:out.size(-2)-pad[1],:],-2)
    
    # Pad Left
    if pad[2] != 0:
        out[...,:,0:pad[2]] = reverse(out[...,:,pad[2]:2*pad[2]],-1)
    
    # Pad Right
    if pad[3] != 0:
        out[...,:,out.size(-1)-pad[3]::] = reverse(out[...,:,out.size(-1)
            -2*pad[3]:out.size(-1)-pad[3]],-1)    
    
    if sflag:
        out.squeeze_()
        
    return out


def symmetricPad_transpose2D(input,crop = 0):
    r"""Adjoint of the SymmetricPad2D operation which amounts to a special type
    of cropping. CROP specifies the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT].
    If crop is an integer then each direction is cropped by the same amount. In
    order to achieve a different amount of cropping in each direction of the 
    tensor, crop needs to be a tuple."""          
    
    # crop = [top,bottom,left,right]
    
    if isinstance(crop,int):
        assert(crop >= 0), """Crop must be either a non-negative integer 
        or a tuple."""
        crop = (crop,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)        
             
    assert(isinstance(crop,tuple) and len(crop) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0), \
            "Crop must be non-negative in each dimension."    
    
    assert (crop[0] + crop[1] <= sz[-2] and crop[2] + crop[3] <= sz[-1]), \
            "Crop does not have valid values."
    
    out = input.clone()
    
    # Top
    if crop[0] != 0:
        out[...,crop[0]:2*crop[0],:] += reverse(out[...,0:crop[0],:],-2)
    
    # Bottom 
    if crop[1] != 0:

        out[...,-2*crop[1]:-crop[1],:] += reverse(out[...,-crop[1]::,:],-2) 
    
    # Left 
    if crop[2] != 0:
        out[...,crop[2]:2*crop[2]] += reverse(out[...,0:crop[2]],-1)
    
    # Right
    if crop[3] != 0:
        out[...,-2*crop[3]:-crop[3]] += reverse(out[...,-crop[3]::],-1) 
    
    if crop[1] == 0:
        end_h = sz[-2]+1 
    else:
        end_h = sz[-2]-crop[1]
        
    if crop[3] == 0:
        end_w = sz[-1]+1
    else:
        end_w = sz[-1]-crop[3]
        
    out = out[...,crop[0]:end_h,crop[2]:end_w]
    
    
    if sflag:
        out.squeeze_()
        
    return out

def pad2D(input,pad=0,padType='zero'):
    r"""Pads the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple. PadType specifies the type of padding.
    Valid padding types are "zero","symmetric" and "periodic". """
    
    pad = formatInput2Tuple(pad,int,4)
    
    if sum(pad) == 0:
        return input
    
    if padType == 'zero':
        return zeroPad2D(input,pad)
    elif padType == 'symmetric':
        return symmetricPad2D(input,pad)
    elif padType == 'periodic':
        return periodicPad2D(input,pad)
    else:
        raise NotImplementedError("Unknown padding type.")
        
def pad_transpose2D(input,pad=0,padType='zero'):
    r"""Transpose operation of pad2D. PAD specifies the amount of padding as 
    [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple. PadType specifies the type of padding.
    Valid padding types are "zero" and "symmetric". """    
    
    pad = formatInput2Tuple(pad,int,4)
    
    if sum(pad) == 0:
        return input
    
    if padType == 'zero':
        return crop2D(input,pad)
    elif padType == 'symmetric':
        return symmetricPad_transpose2D(input,pad)
    elif padType == 'periodic':
        return periodicPad_transpose2D(input,pad)
    else:
        raise NotImplementedError("Uknown padding type.")
        
def formatInput2Tuple(input,typeB,numel,strict = True):
    assert(isinstance(input,(tuple,typeB))),"input is expected to be of type " \
        "tuple or of type " + str(typeB)[8:-2] + " but instead an input of "\
        +"type "+str(type(input))+" was provided."
    
    if isinstance(input,typeB):
        input = (input,)*numel
    
    if strict :
        assert(len(input) == numel), "An input of size "+str(numel)+" is expected "\
            "but instead input = "+str(input)+ " was provided."
    else:
        if len(input) < numel:
            input = input + (input[-1],)*(numel-len(input))
        elif len(input) > numel:
            input = input[0:numel]
        
    return tuple(typeB(i) for i in input)
    
def getPad2RetainShape(kernel_size,dilation = 1):
    r"""Returns the necessary padding in the format [TOP BOTTOM LEFT RIGHT] 
    so that the spatial dimensions of the output will remain the same with 
    the spatial dimensions of the input.
    Note: This function assumes that the conv2d is computed using stride = 1."""
    
    kernel_size = formatInput2Tuple(kernel_size,int,2)
    dilation = formatInput2Tuple(dilation,int,2)
    
    kernel_size = ((kernel_size[0]-1)*dilation[0]+1,(kernel_size[1]-1)*dilation[1]+1)
    Kc = torch.Tensor(kernel_size).add(1).div(2).floor()
    return (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
                                        int(Kc[1])-1,kernel_size[1]-int(Kc[1]))    
    
def reverse(input,dim=0) :
    r"""Reverses the specified dimension of the input tensor."""
    Dims = input.dim()
    assert (dim < Dims), "The selected dimension (arg 2) exceeds the tensor's dimensions."
    idx = torch.arange(input.size(dim)-1,-1,-1).type_as(input).long()
    return input.index_select(dim,idx)

def shift(x,s,bc='circular',value=0):
    """ 
    Shift operator that can treat different boundary conditions. It applies 
    to a tensor of arbitrary dimensions. 
    ----------
    Usage: xs = shift(x,(0,1,-3,3),'reflexive')
    ----------
    Parameters
    ----------
    x : tensor.
    s : tuple that matches the dimensions of x, with the corresponding shifts.
    bc: String with the prefered boundary conditions (bc='circular'|'reflexive'|'zero')
        (Default: 'circular')
    """
    
    if not isinstance(bc, str):
        raise Exception("bc must be of type string")
       
    if not reduce(lambda x,y : x and y, [isinstance(k,int) for k in s]):
        raise Exception("s must be a tuple of ints")
           
    if len(s) < x.dim():
        s = s + (0,) * (x.dim()-len(s))        
    elif len(s) > x.dim():
        print("The shift values will be truncated to match the " \
        +"dimensions of the input tensor. The trailing extra elements will" \
        +" be discarded.")
        s = s[0:x.dim()]
    
    if reduce(lambda x,y : x or y, [ math.fabs(s[i]) > x.shape[i] for i in range(x.dim())]):
        raise Exception("The shift steps should not exceed in absolute values"\
        +" the size of the corresponding dimensions.")

    # use a list sequence instead of a tuple since the latter is an 
    # immutable sequence and cannot be altered         
    indices = [slice(0,x.shape[0])]
    for i in range(1,x.dim()):
        indices.append(slice(0,x.shape[i]))
        
    if bc == 'circular':
        xs = x.clone() # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                m = x.shape[i]
                idx = indices[:]                
                idx[i] = (np.arange(0,m)-s[i])%m
                xs = xs[tuple(idx)]
    elif bc == 'reflexive':
        xs = x.clone() # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                idx = indices[:]
                if s[i] > 0: # right shift                    
                    idx[i] = list(range(s[i]-1,-1,-1)) + list(range(0,x.shape[i]-s[i]))
                else: # left shift
                    idx[i] = list(range(-s[i],x.shape[i])) + \
                    list(range(x.shape[i]-1,x.shape[i]+s[i]-1,-1))
                
                xs = xs[tuple(idx)]
    elif bc == 'zero':
        xs=torch.zeros_like(x)        
        idx_x=indices[:]
        idx_xs=indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:       
                if s[i] > 0: # right shift
                    idx_x[i] = slice(0,x.shape[i]-s[i])
                    idx_xs[i] = slice(s[i],x.shape[i])
                else: # left shift
                    idx_x[i] = slice(-s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]+s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
    elif bc == 'constant':
        xs=torch.ones_like(x)*value
        idx_x=indices[:]
        idx_xs=indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:       
                if s[i] > 0: # right shift
                    idx_x[i] = slice(0,x.shape[i]-s[i])
                    idx_xs[i] = slice(s[i],x.shape[i])
                else: # left shift
                    idx_x[i] = slice(-s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]+s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]        
        
    else:
        raise Exception("Unknown boundary conditions")
    
    return xs

def psf2otf(psf,otfSize):
    r"""Transforms a given 2D psf (point spread function) to a 2D otf (optical
    transfer) function of a specified size"""

    assert(psf.dim() == 2 and len(otfSize) >= 2),"Invalid input for psf and/or otfSize."
    assert(psf.size(0) <= otfSize[-2] and psf.size(1) <= otfSize[-1]),"The "\
    +"spatial support of the otf must be equal or larger to that of the psf."

    otf = torch.zeros(otfSize).type_as(psf)
    otf[...,0:psf.size(0),0:psf.size(1)] = psf

    s = tuple(int(i) for i in -(np.asarray(psf.shape[0:])//2))
    s = (0,)*(len(otfSize)-2)+s
    otf = shift(otf,s,bc='circular')
    otf = torch.rfft(otf,2)

    return otf
               
def imfilter_transpose2D_SpatialDomain(input,kernel,padType="symmetric",mode="conv"):

    assert(mode in ("conv","corr")), "Valid filtering modes are"\
    +" 'conv' and 'corr'."
    assert(padType in ("periodic","symmetric","zero","valid")), "Valid padType"\
    +" values are 'periodic'|'symmetric'|'zero'|'valid'."

    assert(input.dim() < 5),"The input must be at most a 4D tensor."

    while input.dim() <  4:
        input = input.unsqueeze(0)

    while kernel.dim() < 4:
        kernel = kernel.unsqueeze(0)

    channels = input.size(1)
    assert(kernel.size(1) == 1 or kernel.size(1) == channels),"Invalid "\
    +"filtering kernel dimensions."

    if kernel.shape[1] == 1 and input.shape[1] != kernel.shape[1]:
        kernel = torch.cat([kernel]*input.shape[1], dim=1)

    if mode == "conv":
        kernel = reverse(reverse(kernel,dim=-1),dim=-2)

    if padType == "valid":
        padding = 0
    else:
        padding = getPad2RetainShape(kernel.shape[-2:])

    b, c, h, w = input.shape
    input = input.reshape(input.shape[0]*input.shape[1], input.shape[2], input.shape[3])
    input = input[None]
    kernel = kernel.reshape(kernel.shape[0]*kernel.shape[1], kernel.shape[2], kernel.shape[3])

    kernel = kernel[:,None]
    out = torch.conv_transpose2d(input, kernel, groups=kernel.shape[0])
    out = out[0].reshape(b, c, out.shape[2], out.shape[3])
    return pad_transpose2D(out,padding,padType)


def imfilter2D_SpatialDomain(input,kernel,padType="symmetric",mode="conv"):
    r"""If the input and the kernel are both multichannel tensors then each
    channel of the input is filtered by the corresponding channel of the
    kernel.Otherwise, if kernel has a single channel each channel of the input
    is filtered by the same channel of the kernel."""

    assert(mode in ("conv","corr")), "Valid filtering modes are"\
    +" 'conv' and 'corr'."
    assert(padType in ("periodic","symmetric","zero","valid")), "Valid padType"\
    +" values are 'periodic'|'symmetric'|'zero'|'valid'."

    assert(input.dim() < 5),"The input must be at most a 4D tensor."

    while input.dim() <  4:
        input = input.unsqueeze(0)

    while kernel.dim() < 4:
        kernel = kernel.unsqueeze(0)

    channels = input.size(1)
    
    if kernel.shape[1] == 1 and input.shape[1] != kernel.shape[1]:
        kernel = torch.cat([kernel]*input.shape[1], dim=1)
    if mode == "conv":
        kernel = reverse(reverse(kernel,dim=-1),dim=-2)

    if padType == "valid":
        padding = 0
    else:
        padding = getPad2RetainShape(kernel.shape[-2:])

    input = pad2D(input,padding,padType)

    b, c, h, w = input.shape
    input = input.reshape(input.shape[0]*input.shape[1], input.shape[2], input.shape[3])
    input = input[None]
    kernel = kernel.reshape(kernel.shape[0]*kernel.shape[1], kernel.shape[2], kernel.shape[3])
    kernel = kernel[:,None]
    out = torch.conv2d(input, kernel, groups=kernel.shape[0])
    out = out[0].reshape(b, c, out.shape[2], out.shape[3])

    return out       

def odctdict(n,L,dtype = 'f',GPU = False):
    D = torch.zeros(n,L)
    if dtype == 'f':
        D = D.float()
    else:
        D = D.double()
    
    D[:,0] = 1/math.sqrt(n)
    for k in range(1,L): 
        v = torch.cos(th.arange(0,n)*math.pi*k/L); 
        v -= v.mean();
        D[:,k] = v.div(v.norm(p=2))
    
    if torch.cuda.is_available() and GPU:
        D = D.cuda()
    
    return D

def odctndict(n,L,p = None, dtype = 'f', GPU = False):
    r"""  D = ODCTNDICT((N1 N2 ... Np),(L1 L2 ... Lp)) returns an overcomplete 
    DCT dictionary for p-dimensional signals of size N1xN2x...xNp. The number 
    of DCT atoms in the i-th dimension is Li, so the combined dictionary is of
    size (N1*N2*...*Np) x (L1*L2*...*Lp).

    D = ODCTNDICT([N1 N2 ... Np],L) specifies the total number of atoms in
    the dictionary instead of each of the Li's individually. The Li's in
    this case are selected so their relative sizes are roughly the same as
    the relative sizes of the Ni's. Note that the actual number of atoms in
    the dictionary may be larger than L, as rounding might be required for
    the computation of the Li's.

    D = ODCTNDICT(N,L,P) is shorthand for the call ODCTNDICT(N*ones(1,P),L),
    and returns the overcomplete DCT dictionary for P-dimensional signals of
    size NxNx...xN. L is the required size of the overcomplete dictionary,
    and is rounded up to the nearest integer with a whole P-th root.
    """
    assert(isinstance(n,int) or isinstance(n,tuple)), " n should be either of "\
    "type int or of type tuple."
    assert(isinstance(L,int) or isinstance(L,tuple)), " L should be either of "\
    "type int or of type tuple."
    assert(isinstance(p,int) or p is None), " p should be either of "\
    "type int or being omitted."
         
    n = np.asarray(n)
    L = np.asarray(L)

    if p is None:
        p = n.size

    if n.size == 1 :
        n = n*np.ones((1,p))
    if L.size == 1 :
        L = L*np.ones((1,))
        

    if L.size ==1 and p > 1 :
        N = np.prod(n)
        L = np.ceil((L*(np.power(n,p)/N)**(1/(p-1)))**(1/p))
    
    n = tuple(int(i) for i in n)
    L = tuple(int(i) for i in L)
    
    D = odctdict(n[0],L[0],dtype,GPU)
    for i in range(1,p):
        D = kron(D,odctdict(n[i],L[i],dtype,GPU))
    
    return D

def odct2dict(n,L,dtype = 'f', GPU = False):
    return odctndict(n,L,2,dtype,GPU)

def odct3dict(n,L,dtype = 'f', GPU = False):
    return odctndict(n,L,3,dtype,GPU)

def kron(x,y):
    r""" Kronecker tensor product.
    KRON(X,Y) is the Kronecker tensor product of X and Y.
    The result is a large matrix formed by taking all possible
    products between the elements of X and those of Y. For
    example, if X is 2 by 3, then KRON(X,Y) is
 
       [ X[0,0]*Y  X[0,1]*Y  X[0,2]*Y
         X[1,0]*Y  X[1,1]*Y  X[1,2]*Y ]
    """
    assert(x.dim() == 1 or x.dim() == 2), "x must be either a 1D or 2D tensor."
    assert(y.dim() == 1 or y.dim() == 2), "x must be either a 1D or 2D tensor."
    
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    x_size = x.shape
    y_size = y.shape
    
    x = x.t().contiguous().view(-1)
    y = y.t().contiguous().view(-1)
    
    z = y.ger(x)
    
    D = torch.Tensor().type_as(x)
    for m in range(0,x_size[1]):
        d = torch.Tensor().type_as(x)
        for k in range(x_size[0]*m,x_size[0]*(m+1)):
            d = torch.cat((d,z[:,k].contiguous().view(y_size[1],y_size[0]).t()),dim=0)
        if m == 0:
            D = torch.cat((D,d))
        else:
            D = torch.cat((D,d),dim=1)
    
    return D

def dct(tensor):
    """
    Function to initialize the input tensor with weights from the dct basis or dictionary.
    """
    assert (tensor.ndimension() == 4), "A 4D tensor is expected."
    output_features, input_channels, H, W = tensor.shape

    if H * W * input_channels == output_features + 1:
        tensor.data.copy_(gen_dct3_kernel(tensor.shape[1:]).type_as(tensor)[1:, ...])
    else:
        if input_channels == 1:
            weights = odctndict((H, W), output_features + 1)
        else:
            weights = odctndict((H, W, input_channels), output_features + 1)
        weights = weights[:, 1:output_features + 1].type_as(tensor).view(H, W, input_channels, output_features)
        weights = weights.permute(3, 2, 0, 1)
        tensor.data.copy_(weights)

def calc_psnr(inp, other):

    '''
    Function to calculate mean PSNR over a batch
    
    Parameters
    ----------
    input: (torch.(cuda.)Tensor) Restored image of size B x C x H x W
    other: (torch.(cuda.)Tensor) Ground-truth image of size B x C x H x W
    returns: (torch.(cuda.)Tensor) Mean PSNR for a batch
    '''

    assert (torch.is_tensor(inp) and torch.is_tensor(other)), "The first two inputs " \
                                                                + "must be tensors."
    assert (inp.shape == other.shape), "Dimensions mismatch between the two " \
                                         "input tensors."

    while inp.dim() < 4:
        inp = inp.unsqueeze(0)
    while other.dim() < 4:
        other = other.unsqueeze(0)

    PSNR = []

    for i in range(other.shape[0]):
        peakVal = other[i].max()
        MSE = (inp[i] - other[i]).view(-1).pow(2).mean(dim=0)

        SNR = 10 * torch.log10(peakVal ** 2 / MSE)
        PSNR.append(SNR.detach().cpu().numpy())
    
    return np.array(PSNR).mean()


def crop_psf_shape(imgs, kernels):
    """
    Function to crop a batch of 2D images, expanded by boundaries, corresponding to PSF size.

    Parameters
    ----------
    imgs: (torch.(cuda.)Tensor) Padded images of shape B x C x H + Hk -1 x W + Wk -1
    kernels: (torch.(cuda.)Tensor) PSFs of shape B x C x Hk x Wk
    returns: (torch.(cuda.)Tensor) Cropped images of shape B x C x H x W
    """
    crop_updown, crop_leftright = [(i - 1) // 2 for i in kernels.shape[-2:]]
    
    cropped = imgs[:, :, crop_updown:-crop_updown, crop_leftright:-crop_leftright]
    
    return cropped

def anscombe(x):
    '''
    Compute the anscombe variance stabilizing transform.
      the input   x   is noisy Poisson-distributed data
      the output  fx  has variance approximately equal to 1.
    Reference: Anscombe, F. J. (1948), "The transformation of Poisson,
    binomial and negative-binomial data", Biometrika 35 (3-4): 246-254
    https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/anscombe.py
    '''
    return 2.0*torch.sqrt(x + 3.0/8.0)

def exact_unbiased(z):
    '''
    Computes the inverse transform using an approximation of the exact
    unbiased inverse.
    Reference: Makitalo, M., & Foi, A. (2011). A closed-form
    approximation of the exact unbiased inverse of the Anscombe
    variance-stabilizing transformation. Image Processing.  
    https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/anscombe.py
    '''
    return (1.0 / 4.0 * z.pow(2) +
            (1.0/4.0) * math.sqrt(3.0/2.0) * z.pow(-1) -
            (11.0/8.0) * z.pow(-2) +
            (5.0/8.0) * math.sqrt(3.0/2.0) * z.pow(-3) - (1.0 / 8.0))


def pad_psf_shape(imgs, kernels):
    """
    Function to perform a 2D replication padding of images with the size corresponding to PSF size.

    Parameters
    ----------
    imgs: (torch.(cuda.)Tensor) Input images of shape B x C x H x W
    kernels: (torch.(cuda.)Tensor) PSFs of shape B x C x Hk x Wk
    returns: (torch.(cuda.)Tensor) Padded images of shape B x C x H + Hk - 1 x W + Wk - 1
    """    
    
    pad_updown, pad_leftright = [(i - 1) // 2 for i in kernels.shape[-2:]]
    pad_left = pad_right = pad_leftright
    pad_up = pad_down = pad_updown
    padded = nn.ReplicationPad2d((pad_left, pad_right, pad_up, pad_down))(imgs)
    return padded


def cabs(input):
    """
    Returns the pointwise magnitude of the elements of the input complex tensor.
    """

    assert (torch.is_tensor(input) and input.size(-1) == 2), "Inputs is expected " \
                                                          + "to be a complex tensor."

    return input.pow(2).sum(dim=-1).sqrt()

def conj(input):
    """
    Returns the complex conjugate of the input complex tensor.
    """

    assert (torch.is_tensor(input) and input.size(-1) == 2), "Input is expected " \
                                                          + "to be a complex tensor."

    out = input.clone()
    out[..., 1] = -out[..., 1]
    return out


def cmul(input, other):
    """
    Returns the pointwise product of the elements of two complex tensors.
    """

    assert (torch.is_tensor(input) and torch.is_tensor(other)), "Inputs are expected " \
                                                          + "to be tensors."

    assert (input.size(-1) == 2 and other.size(-1) == 2), "Inputs must be " \
                                                          + "complex tensors (their last dimension should be equal to two)."

    real = input[..., 0].mul(other[..., 0]) - input[..., 1].mul(other[..., 1])
    imag = input[..., 0].mul(other[..., 1]) + input[..., 1].mul(other[..., 0])

    return torch.cat((real.unsqueeze(-1), imag.unsqueeze(-1)), dim=-1)


class EdgeTaper(torch.autograd.Function):
    """
    Performs edgetapering on an image before applying a deconvolution algorithm
    """

    @staticmethod
    def forward(ctx, input, psf):
        """
        Parameters
        ----------
        input: (torch.(cuda.)Tensor) Input padded image of shape 1 x C x H + Hk - 1 x W + Wk -1
        psf: (torch.(cuda.)Tensor) Corresponding PSF of shape Hk x Wk
        returns: (torch.(cuda.)Tensor) Edgetapered image of shape 1 x C x H + Hk -1 x W + Wk -1
        """

        assert (input.dim() < 5), "The input is expected to be at most a 4D tensor."
        while input.dim() < 4:
            input = input.unsqueeze(0)

        assert (psf.dim() == 2), "Only 2D psfs are accepted."

        beta = {}

        if psf.size(0) != 1:
            psfProj = psf.sum(dim=1)
            z = torch.zeros(input.size(-2) - 1).type_as(psf)
            z[0:psf.size(0)] = psfProj
            z = torch.rfft(z, 1, onesided=True)
            z = torch.irfft(cmul(z, conj(z)), 1, onesided=True, signal_sizes=(input.size(-2) - 1,))
            z = torch.cat((z, z[0:1]), dim=0).div(z.max())
            beta['dim0'] = z.unsqueeze(-1)

        if psf.size(1) != 1:
            psfProj = psf.sum(dim=0)
            z = torch.zeros(input.size(-1) - 1).type_as(psf)
            z[0:psf.size(1)] = psfProj
            z = torch.rfft(z, 1, onesided=True)
            z = torch.irfft(cmul(z, conj(z)), 1, onesided=True, signal_sizes=(input.size(-1) - 1,))
            z = torch.cat((z, z[0:1]), dim=0).div(z.max())
            beta['dim1'] = z.unsqueeze(0)

        if len(beta.keys()) == 1:
            alpha = 1 - beta[list(beta.keys())[0]]
        else:
            alpha = (1 - beta['dim0']) * (1 - beta['dim1'])

        while alpha.dim() < input.dim():
            alpha = alpha.unsqueeze(0)

        otf = psf2otf(psf, input.shape)

        blurred_input = torch.irfft(cmul(torch.rfft(input, 2), otf), 2, \
                                 signal_sizes=input.shape[-2:])

        output = alpha * input + (1 - alpha) * blurred_input

        if ctx.needs_input_grad[0]:
            mask = torch.__and__(output >= input.min(), output <= input.max())
            ctx.intermediate_results = alpha, otf, mask

        return output.clamp(input.min(), input.max())

    @staticmethod
    def backward(ctx, grad_output):

        alpha, otf, mask = ctx.intermediate_results

        grad_input = mask.type_as(grad_output) * grad_output

        grad_input = alpha * grad_input + torch.irfft(cmul(torch.rfft((1 - alpha) \
                                                   * grad_input, 2), conj(otf)), 2,
                                                   signal_sizes=grad_input.shape[-2:])

        return grad_input, None

def get_image_grad(x):
    '''
    Function for calculation of the gradients of an input along X and Y axes
    
    Parameters
    ----------
    x: (torch.(cuda.)Tensor) Input image B x C x H x W
    return: (torch.(cuda.)Tensor) Tuple of tensors: gradients of an input tensor along X and Y axes
    '''
    weights = torch.Tensor([[-1, 0, 1]])
    if x.is_cuda:
        weights = weights.cuda()
    x_ = F.pad(x, (1, 1, 0, 0), 'reflect')
    grad_x = F.conv2d(x_, torch.stack([weights[None]] * x.shape[1]), groups=x.shape[1])
    x_ = F.pad(x, (0, 0, 1, 1), 'reflect')
    grad_y = F.conv2d(x_, torch.stack([weights.transpose(0, 1)[None]] * x.shape[1]), groups=x.shape[1])
    return grad_x, grad_y
