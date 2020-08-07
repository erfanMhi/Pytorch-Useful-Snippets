import numpy as np

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

def get_same_padding_size(kernel_size=1, stride=1, dilation=1):
    """
    A utility function which calculated the padding size needed to 
    get the same padding functionality as same as tensorflow Conv2D implementation
    """
    neg_padding_size = (stride - dilation*kernel_size + dilation -1)/2
    if neg_padding_size>0:
        return 0
    return int(np.ceil(np.abs(neg_padding_size)))
