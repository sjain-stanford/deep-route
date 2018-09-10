#!/usr/bin/env python

import numpy as np

import random

import os

import h5py




def blockage(input1,x,y,map):
    
    output = input1

    output[map['blockage'], y, x] = 1

    return output



def check(input1,x,y,xpins,ypins,xlen,ylen):
    
    output = input1
    
    xavg = int(np.average(xpins))
    
    yavg = int(np.average(ypins))
    
    xmin = min(xpins)
    xmax = max(xpins)

    ymin = min(ypins)
    ymax = max(ypins)

    if xlen > ylen :
        if y == yavg:
            if (x >= xmin and x <= xmax):
               # print("branchblockage")
                return 0    

        elif y > yavg:
            z1=[]
            for l,m in zip(xpins,ypins):
                if l==x:
                    if m > y: 
                        z1.append(m)

            if z1:
               # print("legblockage")
                return 1
            if not z1:
                return -1

        elif y < yavg :
            z1=[]
            for l,m in zip(xpins,ypins):
                if l==x:
                    if m < y: 
                        z1.append(l)

            if z1:
               # print("legblockage")
                return 1
            if not z1:
                return -1
            
    else:
        if x == xavg:
            if (y >= ymin and y <= ymax):
               # print("branch blockage")
                return 0

        elif x > xavg:
            z1=[]
            for l,m in zip(xpins,ypins):
                if m==y:
                    if l > x: 
                        z1.append(l)

            if z1:
               # print("legblockage")
                return 1
            if not z1:
                return -1

        elif x < xavg :
            z1=[]
            for l,m in zip(xpins,ypins):
                if m==y:
                    if l < x: 
                        z1.append(l)

            if z1:
               # print("legblockage")
                return 1
            if not z1:
                return -1
       
                   





def route_legblockage(input1,x,y,map,xwire,ywire,xlen,ylen,xpins,ypins):

    output = input1

    xavg = int(np.average(xpins))
    
    yavg = int(np.average(ypins))

    if xlen > ylen :
        
        x_min = min(xpins)
        x_max = max(xpins) 

        output[map[xwire], yavg, x_min:x_max+1] = 1
    
        if ywire is not None:
            
            # for get y cordinate of end point of leg 
    
            z1=[]
    
            for m,n in zip(xpins,ypins):
    
                if m==x:
    
                    z1.append(n)

            ymin = min(z1)
            ymax = max(z1)        

            if y < yavg:
    
                output[map[ywire],y+1:yavg+1,x] = 1
    
                output[map[ywire],ymin:y,x] =1
               
            elif y > yavg:
    
                output[map[ywire],yavg:y,x] =1
    
                output[map[ywire],y+1:ymax+1,x] =1   


              #via problem   
    
            if output[map[xwire],y,x+1] !=1:
    
                output[map[ywire],y-1:y+2,x+1] =1 
    
            elif output[map[xwire],y,x+1] ==1:
    
                output[map[ywire],y-1,x+1] =1
    
                output[map[ywire],y+1,x+1] =1


            for n in range(len(xpins)):
                if ypins[n] > yavg:
                    output[map[ywire],yavg:ypins[n]+1, xpins[n]] =1
                else:
                    output[map[ywire], ypins[n]:yavg+1,xpins[n]]=1    
                
        
        if xwire is not None:
            output[map[xwire],y+1,x:x+2] =1
    
            output[map[xwire],y-1,x:x+2] =1


           

    
    else:
        
        y_min = min(ypins)
        y_max = max(ypins)

        output[map[ywire], y_min:y_max+1, xavg] = 1

        
    
        if xwire is not None:

            # for get x cordinate of end point of leg
            z2=[]
            
            for m,n in zip(xpins,ypins):
    
                if n==y:
    
                    z2.append(m)

            xmin= min(z2)
            xmax= max(z2)        
     
            if x < xavg:
        
                output[map[xwire],y,x+1:xavg+1] =1
        
                output[map[xwire],y,xmin:x] =1
        
            elif x > xavg:
        
                output[map[xwire],y,xavg:x] =1
        
                output[map[xwire],y,x+1:xmax+1] =1
   
            #via problem
        
            if output[map[ywire],y+1,x] !=1:
        
                output[map[xwire],y+1,x-1:x+2]=1
        
            elif output[map[ywire],y+1,x] ==1:
        
                output[map[xwire],y+1,x-1]=1
        
                output[map[xwire],y+1,x+1]=1


            for n in range(len(xpins)):
                if xpins[n] > xavg:
                    output[map[xwire],ypins[n],xavg:xpins[n]+1] = 1
                else:
                    output[map[xwire],ypins[n],xpins[n]:xavg+1] = 1        

        if ywire is not None:
            output[map[ywire],y:y+2,x-1]=1
        
            output[map[ywire],y:y+2,x+1]=1
        
        
        
            



    output[map['via3']] = (output[map['m3']] == 1) * (output[map['m4']] == 1) * 1

    output[map['via4']] = (output[map['m4']] == 1) * (output[map['m5']] == 1) * 1

    output[map['via5']] = (output[map['m5']] == 1) * (output[map['m6']] == 1) * 1    
    
    return output
                       



def route_branchblockage(inpuut,x,y,map,xwire,ywire,xlen,ylen,xpins,ypins):
    
    output=inpuut
    
    xavg = int(np.average(xpins))
    
    yavg = int(np.average(ypins))
    
    if xlen > ylen :
    
        xmin = min(xpins)
    
        xmax = max(xpins)

    
        output[map[xwire], yavg, xmin:x] = 1
    
        output[map[xwire], yavg, x+1:xmax+1] = 1
    
        # via problem
    
        if ywire is not  None:
            
            if output[map['pin'],yavg-1,x] == 1:
               
                output[map[ywire],yavg-1:yavg+1,x-1]=1
               
                output[map[ywire],yavg-1:yavg+1,x+1]=1
                

                if output[map[ywire],y-1,x] !=1:
               
                    output[map[xwire],y-1,x-1:x+2] =1
               

                elif output[map[ywire],y-1,x] == 1:
               
                    output[map[xwire],y-1,x-1]=1
               
                    output[map[xwire],y-1,x+1]=1
                


            else:
                
                output[map[ywire], yavg:yavg+2,x-1]=1
                
                output[map[ywire], yavg:yavg+2,x+1]=1
                
    
                if output[map[ywire],y+1,x] != 1:
    
                    output[map[xwire],y+1,x-1:x+2] =1
    

                elif output[map[ywire],y+1,x] ==1 :
    
                    output[map[xwire],y+1,x-1] =1
    
                    output[map[xwire],y+1,x+1] =1


            
            z3=[]
            z4=[] # contain ypins
            z5=[] # contain xins
            
            for a,b in zip(xpins,ypins):
            	
            	if a == x:

                   z3.append(b)
                
                elif a != x:
                	z4.append(b)

            
            if z3:
                temp1 = min(z3)
                temp2 = max(z3)

                output[map[ywire],yavg+1:temp2+1,x] =1
           
                output[map[ywire],temp1:yavg,x] =1 

            for c,d in zip(xpins,ypins):

                if d in z4 :
                	z5.append(c)
            
            for f in range(len(z4)):
            	if z4[f] > yavg:
            		output[map[ywire],yavg:z4[f]+1, z5[f]] =1

            	else:
            		output[map[ywire],z4[f]:yavg+1, z5[f]] =1

                          

    
        if ywire is None:
    
            ylen=1
    
            xwire, ywire = selectWireClass(xlen, ylen)
    
            #output[map[ywire],y,x]=2
    
            output[map[ywire], yavg:yavg+2,x-1] =1
    
            output[map[ywire], yavg:yavg+2,x+1] =1            



    else:
    
        ymin = min(ypins)
    
        ymax = max(ypins)

        output[map[ywire],ymin:y,xavg]=1
    
        output[map[ywire],y+1:ymax+1,xavg]=1
    
        # for via problem
    
        if xwire is not None:
           # output[map[xwire],y,x] =2
            if output[map['pin'],y,xavg-1]==1:
                output[map[xwire],y-1,xavg-1:xavg+1]=1
                output[map[xwire],y+1,xavg-1:xavg+1]=1

                if output[map[xwire],y,x-1] !=1:
                    output[map[ywire],y-1:y+2,x-1]=1
                elif output[map[xwire],y,x-1] ==1:
                    output[map[ywire],y-1,x-1]=1
                    output[map[ywire],y+1,x-1]=1    
            
            else:


                output[map[xwire],y-1,xavg:xavg+2]=1
                output[map[xwire],y+1,xavg:xavg+2]=1
    
                if output[map[xwire],y,x+1] !=1:
    
                    output[map[ywire],y-1:y+2,x+1] =1
    
                elif output[map[xwire],y,x+1] == 1:
    
                    output[map[ywire],y-1,x+1]=1
    
                    output[map[ywire],y+1,x+1]=1


            za=[]

            zx=[] # contain xpins

            zy=[] # contain yins
            
            for a,b in zip(xpins,ypins):
            	
            	if b == y:

                   za.append(a)
                
                elif b != y:
                	zx.append(a)

            if za :
                temp1 = min(za)
                temp2 = max(za)

                output[map[xwire],y,xavg+1:temp2+1] =1
           
                output[map[xwire],y,temp1:xavg] =1 

            for c,d in zip(xpins,ypins):

                if c in zx :
                	zy.append(d)
            
            for f in range(len(zx)):
            	if zx[f] > xavg:
            		output[map[xwire],zy[f],xavg:zx[f]+1] =1

            	else:
            		output[map[xwire],zy[f],zx[f]:xavg+1] =1                

    
        if xwire is None:
    
           xlen =1
    
           xwire , ywire = selectWireClass(xlen,ylen)
    
           #output[map[xwire],y,x] =2
    
           output[map[xwire],y-1,xavg:xavg+2]=1
    
           output[map[xwire],y+1,xavg:xavg+2]=1            




    output[map['via3']] = (output[map['m3']] == 1) * (output[map['m4']] == 1) * 1

    output[map['via4']] = (output[map['m4']] == 1) * (output[map['m5']] == 1) * 1

    output[map['via5']] = (output[map['m5']] == 1) * (output[map['m6']] == 1) * 1
            
    return output                







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

                  [pin, m3, via3, m4, via4, m5, via5, m6, blockage])

                  

         X[:, 0, :, :]  pin (m2)

         

         Y[:, 0, :, :]  pin (m2) - same as X[:, 0, :, :]

         Y[:, 1, :, :]  m3 (vert)

         Y[:, 2, :, :]  via3

         Y[:, 3, :, :]  m4 (horz)

         Y[:, 4, :, :]  via4

         Y[:, 5, :, :]  m5 (vert)

         Y[:, 6, :, :]  via5

         Y[:, 7, :, :]  m6 (horz)

         Y[:, 8, :, :]  blockage

    """



    # 9 layout layers for now

    C = 9



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

        
        # Blockage

        'blockage' : 8,

        
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

        xavg = int(np.average(x_pins))
    
        yavg = int(np.average(y_pins))

        xmin = min(x_pins)

        xmax = max(x_pins)

        ymin = min(y_pins)

        ymax = max(y_pins)


        max_xlen = (max(x_pins) - min(x_pins)) * microns_per_xpixel   # length in um

        max_ylen = (max(y_pins) - min(y_pins)) * microns_per_ypixel   # length in um

        

        # Corner case when pins overlap each other (invalid case)

        # Bug fix for https://github.com/sjain-stanford/RouteAI/issues/4

        if (max_xlen == 0) and (max_ylen == 0):

            continue



        # Draw pins on layer 'pin (m2)' of both X (data) and Y (labels)
           
        
        #if(max_xlen > max_ylen):
        
           # print("........max_xlen:%d------"%(max(x_pins)-min(x_pins)))
        
        #else:
        
           # print("--------max_ylen:%d------"%(max(y_pins)-min(y_pins)))    


        X[n] = drawPins(X[n], x_pins, y_pins, l_map)
        
        Y[n] = drawPins(Y[n], x_pins, y_pins, l_map)

    
        # Add routes to Y (labels)

        x_wire, y_wire = selectWireClass(max_xlen, max_ylen)
     
        #print("--------x_wire,y_wire--------------")
     
        #print(x_wire,y_wire)
        #print(zip(x_pins,y_pins))

        #Y[n] = drawRoutes(Y[n], x_pins, y_pins, max_xlen, max_ylen, x_wire, y_wire, l_map)    
     
        #print("-----------------Y[n]-----------")
     
        #print(Y[n])
        range1 = (0,31) # for blockage
       
        range2 = (xmin,xmax+1)
       
        range3 = (ymin,ymax+1)
       
        rand = (0,3)
       
        k=np.random.randint(*rand)

        # x and y are cordinates for blockage

        # for getting information about leg and branch class you may take reference of research paper 
        
        if k==0:
            
            # k = 0 means blockage can be anywhere 

            x= np.random.randint(*range1)
       
            y= np.random.randint(*range1)
       
        elif k==1:

        	# k = 1 means we set the blockage on branch class
       
            if max_xlen > max_ylen:
                    #--------- horizontal branch blockage---------------
                 
                for j in range(0,100):
       
                    i=np.random.randint(*range2)
       
                    if (i != 31 and i not in x_pins):
                       
                        x=i
                       
                        break
                    
                    else:
                       
                        continue    
                
                y=yavg
                
            else:
                    #--------- vertical branch blockage-------------------
                    
                x=xavg
                
                for j in range(0,100):
                    
                    i=np.random.randint(*range3)
                    
                    if (i !=31 and i not in y_pins):
                    
                        y=i
                    
                        break
                    
                    else:
                    
                        continue    
                
        elif k==2:

        	# k = 2 means We set the blockage on leg class
            
            if max_xlen > max_ylen:
                   

                    # -------------vertical leg blockage-------------
                
                for j in range(0,100):
                
                    i=np.random.choice(x_pins)
                
                    if i != 31 :
                
                        x=i
                
                        break
                
                    else:
                
                        continue    
    
                
                z1=[]
                
                for i,j in zip(x_pins,y_pins):
                    
                    if i==x:
                    
                        z1.append(j)

                
                temp1=min(z1)
                
                temp2=max(z1) 

                if (temp1 - yavg) > 1:
                
                    l=list(range(yavg+1,temp2))
                    
                    range4=[t for t in l if t not in z1]
                
                    for j in range(0,100):
                
                        i=np.random.choice(range4)
                
                        if i != 31:
                
                            y=i
                
                            break
                
                        else:
                
                            continue    
                    

                elif (yavg - temp2) > 1 :
                    
                    l=list(range(temp1+1,yavg))
                
                    range4=[t for t in l if t not in z1]
                
                    for j in range(0,100):
                
                        i=np.random.choice(range4)
                
                        if i != 31:
                
                            y=i
                
                            break
                
                        else:
                
                            continue    
                    

                elif (temp2 - yavg) > 1 and  (yavg - temp1) > 1:
                
                    l=list(range(temp1+1,temp2))
                
                    range4=[t for t in l if t not in z1]
                
                    for j in range(0,100):
                
                        i=np.random.choice(range4)
                
                        if i != 31:
                
                            y=i
                
                            break
                
                        else:
                
                            continue    
                    
                else:
                
                    for b in range(0,100):
                
                        z=np.random.randint(*range1)
                
                        if z != yavg:
                
                            y=z
                
                            break
                
                        else:
                
                            continue    


                    

            else:
                    #------------horizontal leg blockage--------------------
                
                for j in range(0,100):
                
                        i=np.random.choice(y_pins)
                
                        if i != 31:
                
                            y=i
                
                            break
                
                        else:
                
                            continue    

                
                z2=[]
                
                for i,j in zip(x_pins,y_pins):
                
                    if j==y:
                
                        z2.append(j)

                
                temp1=min(z2)
                
                temp2=max(z2)


                        
                if (temp1 - xavg) > 1 :
                    
                    l=list(range(xavg+1,temp2))
                
                    range4=[t for t in l if t not in z2]
                
                    for j in range(0,100):
                
                        i=np.random.choice(range4)
                
                        if i != 31:
                
                            x=i
                
                            break
                
                        else:
                
                            continue    
                    
                    
                elif (xavg - temp2) > 1:
                    
                    l=list(range(temp1+1,xavg))
                
                    range4=[t for t in l if t not in z2]
                
                    for j in range(0,100):
                
                        i=np.random.choice(range4)
                
                        if i != 31:
                
                            x=i
                
                            break
                
                        else:
                
                            continue    
                    
                elif (temp2 - xavg) > 1 and (xavg - temp1) > 1 :
                
                    l=list(range(temp1+1,temp2))
                
                    range4=[t for t in l if t not in z2]
                
                    for j in range(0,100):
                
                        i=np.random.choice(range4)
                
                        if i != 31:
                
                            x=i
                
                            break
                
                        else:
                
                            continue    
                    
                else:
                    for b in range(0,100):
                
                        z=np.random.randint(*range1)
                
                        if z != xavg:
                
                            x=z
                
                            break
                
                        else:
                
                            continue





        
        
        # To set the value of blockage 1 in Y[n] input
        Y[n] = blockage(Y[n], x, y,l_map)


        # To check the position of blockage where it is...
        k = check(Y[n],x,y,x_pins,y_pins,max_xlen,max_ylen)

        # if k=0 : it means there is no blockage on routing path
        # if k=1 : it means there is a blockage in leg class
        # if k=-1 : it means there is a blockage in branch class

       
        if k==0:
       
            Y[n]=route_branchblockage(Y[n],x,y,l_map,x_wire,y_wire,max_xlen,max_ylen,x_pins,y_pins)
       
           # print("-----------------branchblockage route-----------")
       
           # print(Y[n])
        
        elif k ==1:

            Y[n]=route_legblockage(Y[n], x, y, l_map, x_wire, y_wire, max_xlen, max_ylen, x_pins, y_pins)
           # print("-----------------routelegblockage-----------")
           # print(Y[n])


           
        elif k== -1:

            Y[n] = drawRoutes(Y[n], x_pins, y_pins, max_xlen, max_ylen, x_wire, y_wire, l_map)
           # print("-----------------route-----------")
           # print(Y[n])
         
        


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