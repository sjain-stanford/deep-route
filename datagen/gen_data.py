#!/usr/bin/env python
import numpy as np
import random
import os
import h5py


def drawPins(input_data, xpins, ypins, map):
    """
    Draw pins on input_data
    
    Inputs:
       - input_data: np.array of size (C, H, W) (all zeros to start with)
       - xpins & ypins: np.array of x-coordinates and y-coordinates for all pins
                    e.g., [x1, x2 ... xm] and [y1, y2, ... ym] for m pins
       - map: layout layer map (dictionary from layer name to index)
                    
    Outputs:
       - output_data: np.array of size (C, H, W) (with pixels corresponding to pins in layer 'pins' set to '1')
    """
    
    output_data = input_data
    
    for (x, y) in zip(xpins, ypins):
        output_data[map['pin'], y, x] = 1
    
    return output_data


def drawRoutes(input_data, xpins, ypins, xlen, ylen, xwire, ywire, map):
    """
    Draw routes on input_data
    
    Inputs:
       - input_data: np.array of size (C, H, W)
       - xpins & ypins: np.array of x-coordinates and y-coordinates for all pins
                    e.g., [x1, x2 ... xm] and [y1, y2, ... ym] for m pins
       - xlen & ylen: horizontal and vertical max length (in um)
       - xwire & ywire: horizontal and vertical wire class to route; such as ('m4', 'm3')
       - xavg & yavg: horizontal and vertical average points for placing branch of route
       - map: layout layer map (dictionary from layer name to index)
       
    Outputs:
       - output_data: np.array of size (C, H, W)
    """
    
    output_data = input_data
    
    xavg = int(np.average(xpins))
    yavg = int(np.average(ypins))
    
    if xlen > ylen:
        # Draw horizontal branch first
        xmin = min(xpins)
        xmax = max(xpins)
        output_data[map[xwire], yavg, xmin:xmax+1] = 1
        
        # Draw vertical legs from all pins to branch
        # Except in the special case of nPins=2 and pins being x-aligned, where only horizontal branch is needed
        if ywire is not None:
            for n in range(len(xpins)):
                if ypins[n] > yavg:
                    output_data[map[ywire], yavg:ypins[n]+1, xpins[n]] = 1
                else:
                    output_data[map[ywire], ypins[n]:yavg+1, xpins[n]] = 1
    else:            
        # Draw vertical branch first
        ymin = min(ypins)
        ymax = max(ypins)
        output_data[map[ywire], ymin:ymax+1, xavg] = 1
        
        # Draw horizontal legs from all pins to branch
        # Except in the special case of nPins=2 and pins being y-aligned, where only vertical branch is needed
        if xwire is not None:
            for n in range(len(xpins)):
                if xpins[n] > xavg:
                    output_data[map[xwire], ypins[n], xavg:xpins[n]+1] = 1
                else:
                    output_data[map[xwire], ypins[n], xpins[n]:xavg+1] = 1
    
    # Draw vias
    output_data[map['via3']] = (output_data[map['m3']] == 1) * (output_data[map['m4']] == 1) * 1
    output_data[map['via4']] = (output_data[map['m4']] == 1) * (output_data[map['m5']] == 1) * 1
    output_data[map['via5']] = (output_data[map['m5']] == 1) * (output_data[map['m6']] == 1) * 1
    
    return output_data


def selectWireClass(xlen, ylen):
    """
    Draw pins on input_data
    
    Inputs:
       - xlen & ylen: horizontal and vertical max length (in um)
              
    Outputs:
       - wire_class: tuple of (xwire, ywire); such as ('m4', 'm3')
    """
    # 7FF Wire Performance (from RCCalc)
    # R = rho * l / (t * w)
    # rho / (t * w) = R / l
    # Multiply rho_tw by length to get wire resistance, add via resistance to this

    # m2 1x w=20nm horz
    rho_tw_m2_1x = 106.45 # ohm/um
    # m3 2x w=40nm vert
    rho_tw_m3_2x = 30.555  # ohm/um
    # m4 1x w=40nm horz
    rho_tw_m4_1x = 24.077  # ohm/um
    # m5 1x w=38nm vert
    rho_tw_m5_1x = 18.368  # ohm/um
    # m6 1x w=40nm horz
    rho_tw_m6_1x = 16.950  # ohm/um
    # m7 1x w=76nm vert
    rho_tw_m7_1x = 8.854   # ohm/um

    # via resistance
    r_via2 = 40  # ohm/ct
    r_via3 = 30  # ohm/ct
    r_via4 = 12  # ohm/ct
    r_via5 = 12  # ohm/ct
    r_via6 = 12  # ohm/ct
    
    # Assumption - All routes run to/from two m2 pins
    R_m2 = (rho_tw_m2_1x * xlen)
    R_m3 = (rho_tw_m3_2x * ylen) + (r_via2 * 2)
    R_m4 = (rho_tw_m4_1x * xlen) + (r_via2 * 2) + (r_via3 * 2)
    R_m5 = (rho_tw_m5_1x * ylen) + (r_via2 * 2) + (r_via3 * 2) + (r_via4 * 2)
    R_m6 = (rho_tw_m6_1x * xlen) + (r_via2 * 2) + (r_via3 * 2) + (r_via4 * 2) + (r_via5 * 2)
    R_m7 = (rho_tw_m7_1x * ylen) + (r_via2 * 2) + (r_via3 * 2) + (r_via4 * 2) + (r_via5 * 2) + (r_via6 * 2)
    
    # Choose the less resistant wire class based on whether vert or horz length is dominant
    if xlen > ylen:
        if R_m4 > R_m6:
            xwire = 'm6'
            ywire = 'm5'
        else:
            xwire = 'm4'
            if R_m3 > R_m5:
                ywire = 'm5'
            else:
                ywire = 'm3'
    else:
        if R_m5 > R_m3:
            ywire = 'm3'
            xwire = 'm4'
        else:
            ywire = 'm5'
            if R_m4 > R_m6:
                xwire = 'm6'
            else:
                xwire = 'm4'
    
    # Special case: 2 pins that are either x-aligned or y-aligned
    if xlen == 0:
        xwire = None
    if ylen == 0:
        ywire = None
        
    wire_class = (xwire, ywire)
    
    return wire_class
    

def genData(N=10, H=32, W=32, pin_range=(2, 6)):
    """
    Generates decoded training image dataset of size (N, C, H, W)
    
    Inputs:
       - N: number of images to generate
       - H, W: image height, width (in px)
       - pin_range: tuple (low, high) for allowed range of number of pins; "half-open" interval [low, high)
                      (e.g. (2, 6) pins means 2 or 3 or 4 or 5 pins)
                    
    Outputs:
       - Saves X: training data (N, 1, H, W)
       - Saves Y: training labels (N, C, H, W)
             where C = 8 (each corresponds to a layout layer, viz.
                  [pin, m3, via3, m4, via4, m5, via5, m6])
                  
         X[:, 0, :, :]  pin (m2)
         
         Y[:, 0, :, :]  pin (m2) - same as X[:, 0, :, :]
         Y[:, 1, :, :]  m3 (vert)
         Y[:, 2, :, :]  via3
         Y[:, 3, :, :]  m4 (horz)
         Y[:, 4, :, :]  via4
         Y[:, 5, :, :]  m5 (vert)
         Y[:, 6, :, :]  via5
         Y[:, 7, :, :]  m6 (horz)
    """

    # 8 layout layers for now
    C = 8

    data_dir = os.getcwd() + "/data/"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".hdf5"):
                os.remove(data_dir+file)
    else:
        os.makedirs(data_dir)

    data = h5py.File(data_dir + "layout_data.hdf5")

    # numpy arrays no longer needed; use HDF5 instead
    #X = np.zeros([N, 1, H, W], dtype = np.int8)
    #Y = np.zeros([N, C, H, W], dtype = np.int8)

    X = data.create_dataset("X", shape=(N, 1, H, W), dtype='uint8', compression='lzf', chunks=(1, 1, H, W))
    Y = data.create_dataset("Y", shape=(N, C, H, W), dtype='uint8', compression='lzf', chunks=(1, 1, H, W))
    
    # Set physical size represented by HxW pixels
    microns = 11.0    # To have balanced dataset covering from m3 to m6 (based on resistance plots from resistance_vs_distance.ipynb)
    microns_per_xpixel = microns/W
    microns_per_ypixel = microns/H
    
    # Layer map
    l_map = {
        # Pins
        'pin' : 0,
        
        # Vias
        'via3' : 2,
        'via4' : 4,
        'via5' : 6,
        
        # Vertical tracks
        'm3' : 1,
        'm5' : 5,
      
        # Horizontal tracks
        'm4' : 3,
        'm6' : 7
    }
    
    #m3_m4 = m5_m4 = m5_m6 = 0
    
    n = 0
    print_every = 5000
    
    while n < N:
        # Randomly select number of pins from given range
        # Uniform distribution over pin range
        nPins = np.random.randint(*pin_range)
        # Non-uniform distribution (skewed exponentially towards smaller number of pins)
        #p_range = np.array(range(*pin_range))
        #p = np.exp(-p_range) / np.sum(np.exp(-p_range))
        #nPins = np.random.choice(p_range, p=p)
        
        # Randomly pick x and y co-ords for nPins from [0, W) and [0, H) pixels
        x_pins = np.random.randint(W, size=nPins)
        y_pins = np.random.randint(H, size=nPins)
        
        max_xlen = (max(x_pins) - min(x_pins)) * microns_per_xpixel   # length in um
        max_ylen = (max(y_pins) - min(y_pins)) * microns_per_ypixel   # length in um
        
        # Corner case when pins overlap each other (invalid case)
        # Bug fix for https://github.com/sjain-stanford/RouteAI/issues/4
        if (max_xlen == 0) and (max_ylen == 0):
            continue

        # Draw pins on layer 'pin (m2)' of both X (data) and Y (labels)
        X[n] = drawPins(X[n], x_pins, y_pins, l_map)
        Y[n] = drawPins(Y[n], x_pins, y_pins, l_map)
        
        # Add routes to Y (labels)
        x_wire, y_wire = selectWireClass(max_xlen, max_ylen)
        Y[n] = drawRoutes(Y[n], x_pins, y_pins, max_xlen, max_ylen, x_wire, y_wire, l_map)
        
        n += 1
        
        if (n % print_every == 0):
            print("Finished generating %d samples." %(n))
        
        #if x_wire == 'm4' and y_wire == 'm3':
        #    m3_m4 += 1
        #elif x_wire == 'm4' and y_wire == 'm5':
        #    m5_m4 += 1
        #elif x_wire == 'm6' and y_wire == 'm5':
        #    m5_m6 += 1
        #else:
        #    print(x_wire, y_wire)
        
    #print(m3_m4, m5_m4, m5_m6)
    
    # Storing as .npy using np.save -> Issue: RAM out of memory, disk memory limitation   
    #data_dir = os.getcwd() + '/data/'    
    #if os.path.exists(data_dir):
    #    for file in os.listdir(data_dir):
    #        if file.endswith(".npy"):
    #            os.remove(data_dir+file)
    #else:
    #    os.makedirs(data_dir)
    #X_save = data_dir + 'X_save.npy'
    #Y_save = data_dir + 'Y_save.npy'    
    #np.save(X_save, X, allow_pickle=False)
    #np.save(Y_save, Y, allow_pickle=False)
    
    print("Dataset generated as follows:")
    for ds in data:
        print(ds, data[ds])

        
############################
#           MAIN           #
############################
def main():
    N = int(input('Enter the number of images to be generated: '))
    genData(N)


if __name__ == '__main__':
    main()