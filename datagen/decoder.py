import numpy as np
import os
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def decodeData(inp):
    """
    Generates decoded (RGB) images from 8 channel data
    
    Input:
       - inp: data of shape (N, C, H, W) (range: 0 or 1)
             where C = 8 (Y) or 1 (X)
                    
    Output:
       - out: decoded (RGB) data (N, 3, H, W) (range: 0-255)
    """

    # Define RGB colors (http://www.rapidtables.com/web/color/RGB_Color.htm)
    M6_clr    = np.array([[  0.0, 191.0, 255.0]]) # layer 7  m6    dodger blue
    VIA5_clr  = np.array([[255.0,   0.0,   0.0]]) # layer 6  via5  red
    M5_clr    = np.array([[169.0, 169.0, 169.0]]) # layer 5  m5    dark gray
    VIA4_clr  = np.array([[  0.0, 255.0,   0.0]]) # layer 4  via4  green
    M4_clr    = np.array([[255.0,  99.0,  71.0]]) # layer 3  m4    tomato red
    VIA3_clr  = np.array([[  0.0,   0.0, 255.0]]) # layer 2  via3  blue
    M3_clr    = np.array([[ 50.0, 205.0,  50.0]]) # layer 1  m3    lime green
    PIN_clr   = np.array([[  0.0,   0.0,   0.0]]) # layer 0  pin   black
    BGND_clr  = np.array([[255.0, 255.0, 255.0]]) # background     white

    # If inp is 1 image, reshape
    if (len(inp.shape) == 3):
        C, H, W = inp.shape
        inp = inp.reshape(1, C, H, W)

    N, C, H, W = inp.shape
    D = N * H * W

    # Warn if any input elements are out of range
    if np.any(np.greater(inp, 1)):
        print("Invalid input range detected (some input elements > 1)")
    if np.any(np.less(inp, 0)):
        print("Invalid input range detected (some input elements < 0)")

    # Initialize pseudo output & activepixelcount matrix
    out = np.zeros([N, 3, H, W])

    inp_swap = np.swapaxes(inp, 1, 0).reshape(C, D)   # (C, N*H*W)
    out_swap = np.swapaxes(out, 1, 0).reshape(3, D)   # (3, N*H*W)
    
    if C != 1:
        #layer 7 processing
        temp7 = np.broadcast_to((inp_swap[7, :] == 1), (3, D))
        out_swap += M6_clr.T * temp7

        #layer 6 processing
        temp6 = np.broadcast_to((inp_swap[6, :] == 1), (3, D)) 
        out_swap += VIA5_clr.T * temp6

        #layer 5 processing
        temp5 = np.broadcast_to((inp_swap[5, :] == 1), (3, D))
        out_swap += M5_clr.T * temp5

        #layer 4 processing
        temp4 = np.broadcast_to((inp_swap[4, :] == 1), (3, D))
        out_swap += VIA4_clr.T * temp4

        #layer 3 processing
        temp3 = np.broadcast_to((inp_swap[3, :] == 1), (3, D))
        out_swap += M4_clr.T * temp3

        #layer 2 processing
        temp2 = np.broadcast_to((inp_swap[2, :] == 1), (3, D))
        out_swap += VIA3_clr.T * temp2

        #layer 1 processing
        temp1 = np.broadcast_to((inp_swap[1, :] == 1), (3, D))
        out_swap += M3_clr.T * temp1

    #layer 0 processing
    temp0 = np.broadcast_to((inp_swap[0, :] == 1), (3, D))
    out_swap += PIN_clr.T * temp0
    
    # For every pixel, count the number of active layers that overlap
    overlap = 1.0 * np.sum(inp_swap, axis=0, keepdims=True)
    n_layers = overlap + (overlap == 0.0) * 1.0    # To avoid division by zero

    # If no active layers, fill in white
    temp_white = np.broadcast_to((overlap == 0), (3, D))
    out_swap += BGND_clr.T * temp_white

    # Average the colors for all active layers
    out_swap /= (n_layers * 1.0)
    
    out_swap = out_swap.reshape(3, N, H, W)
    out = np.swapaxes(out_swap, 1, 0)    # (N, 3, H, W)

    # Warn if any output elements are out of range
    if np.any(np.greater(out, 255)):
        print ("Invalid output range detected (some output elements > 255)")
    if np.any(np.less(out, 0)):
        print ("Invalid output range detected (some output elements < 0)")

    return out

  
############################
#           MAIN           #
############################
def main():
  plt.rcParams['figure.figsize'] = (15.0, 10.0) # set default size of plots
  plt.rcParams['image.interpolation'] = 'nearest'
  plt.rcParams['image.cmap'] = 'gray'

  data = {}
  data_file = os.getcwd() + "/data/layout_data.hdf5"

  #data = h5py.File(data_file, 'r')
  #X = np.array(data['X'])   # (N, 1, H, W)
  #Y = np.array(data['Y'])   # (N, C, H, W)

  with h5py.File(data_file, 'r') as f:
      for k, v in f.items():
          data[k] = np.asarray(v)

  X_dec = decodeData(data['X'])       # (N, 3, H, W)
  X_dec = np.swapaxes(X_dec, 1, 2)    # (N, H, 3, W)
  X_dec = np.swapaxes(X_dec, 2, 3)    # (N, H, W, 3)

  Y_dec = decodeData(data['Y'])       # (N, 3, H, W)
  Y_dec = np.swapaxes(Y_dec, 1, 2)    # (N, H, 3, W)
  Y_dec = np.swapaxes(Y_dec, 2, 3)    # (N, H, W, 3)

  num_images = 6

  for n in range(num_images):
      plt.subplot(2, 3, n+1)
      plt.imshow(X_dec[n].astype('uint8'))
      #plt.axis('off')
      plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.0)

  plt.savefig("data/sample_X.jpg")
  plt.savefig("data/sample_X.eps")

  for n in range(num_images):
      plt.subplot(2, 3, n+1)
      plt.imshow(Y_dec[n].astype('uint8'))
      #plt.axis('off')
      plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.0)

  plt.savefig("data/sample_Y.jpg")
  plt.savefig("data/sample_Y.eps")

  
if __name__ == '__main__':
  main()
