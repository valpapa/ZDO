import xmltodict
import pprint
import json
from matplotlib import pyplot as plt
import numpy as np


import skimage
import skimage.io
import skimage.color
import skimage.morphology
import numpy
from skimage.filters import threshold_local

from skimage import io, filters


import numpy as np
from skimage import io, morphology, transform
from skimage.transform import probabilistic_hough_line
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math

import metody
import json
import argparse


def prog(name_out,visual_mode=False,  *pics):    #visual_mode=False,
    data=[]
    obr1=[]
    maxp1=[]
    maxl1=[]
    bod11=[]
    bod21=[]
    pojistka=[]
    for path in pics:
        #plt.figure()
        text_color = skimage.io.imread(path)
        obr1.append(text_color)
        #plt.imshow(text_color)
        text_gray = skimage.color.rgb2gray(text_color)
        textb = text_gray > 0.35

        block_size = 45
        adaptive_threshold = threshold_local(text_gray, block_size, offset=0.05)


        textb=text_gray>adaptive_threshold



        size=len(text_color[1])//120
        #print(size)
        kernel_big = skimage.morphology.diamond(size)
        kernel = np.ones((5, 1), dtype=np.uint8)

        textd = skimage.morphology.binary_closing(textb, kernel_big)  #erosion dilation opening closing
#plt.figure()
#plt.imshow(textd, cmap='gray')
        #plt.figure()
        #plt.imshow(textd)
        kernel = np.ones((1, 5), dtype=np.uint8)
#kernel=skimage.morphology.diamond(3)
        #textd = skimage.morphology.binary_dilation(textd, kernel)


        #plt.figure()
        #plt.imshow(textd)
#print(kernel)


#plt.figure()
        skela = skimage.morphology.skeletonize( np.logical_not(textd))
#plt.imshow(skela)


        image=skela
        #plt.figure()
        #plt.imshow(image)
#binary_image=np.logical_not(image)
        binary_image=image
        #plt.figure()

        thresholded_image=binary_image



        X=[]
        Y=[]
        points=[]
        for x in range(0,len(binary_image)):
            for y in range(0,len(binary_image[0])):
                if binary_image[x][y]==True:
                    points+=[(y,x)]
                    X.append(y)
                    Y.append(x)
            
#print(binary_image[1][:])
#print(points)
#plt.figure()
        #plt.plot(X,Y,'.')
        #plt.xlim(0, len(binary_image[0]))
        #plt.ylim( len(binary_image),0)
#for x in range(0,len(points)):
 #   plt.plot(X,Y)

        p=0
        x=0
        
        if len(points)==0:
            del obr1[-1]            
            continue
            
        while True:
            if   len(text_color)-len(text_color)//8 < points[x][1] or points[x][1]< len(text_color)//8 or len(text_color[1])-len(text_color[1])//12 <points[x][0] or points[x][0]< len(text_color[1])//12:
                points.remove(points[x]) 
                p=0
            x=x+1        
            if x>=len(points):
                if p==1:
                    break
                p=1   
                x=0   
        


        
        maxl=(1000,1000)
        maxp=(0,0)
        bod1=[]
        bod2=[]
        
        stred=4
        for point in points:
            if maxl[0]>point[0] and len(text_color)/2+len(text_color)//stred >point[1]> len(text_color)/2-len(text_color)//stred:
                maxl=point
            if maxp[0]<point[0] and len(text_color)/2+len(text_color)//stred>point[1]> len(text_color)/2-len(text_color)//stred:
                maxp=point
			
        
        maxp1.append(maxp)
        maxl1.append(maxl)
        #plt.figure()
        #plt.plot([maxp[0],maxl[0]],[maxp[1],maxl[1]], linestyle='-', color='blue')
        #plt.xlim(0, len(binary_image[0]))
        #plt.ylim( len(binary_image),0)
     

        rozptyl=10
        pomocny=0      
        points2= points[:]
        while True:
            if pomocny==1:
                break
    
            if len(bod1)>12:
                bod1=[]
                bod2=[]
                points=points[:]
                rozptyl=rozptyl+5
            max_value = max(points, key=lambda x: x[1])
            
            if abs(max_value[1]-maxl[1])<rozptyl and abs(maxl[0]-max_value[0])<abs(maxp[0]-max_value[0]):
                points.remove(max_value)
                if len(points)==0:
                    pomocny=1
                    break
                max_value = max(points, key=lambda x: x[1])
                continue
                
     
                
        
            if abs(max_value[1]-maxp[1])<rozptyl and  abs(maxl[0]-max_value[0])>abs(maxp[0]-max_value[0]):
         
                break    
    
            min_value=(1000,1000)
            for point in points:
                if point[1]<min_value[1] and abs(point[0]-max_value[0])<rozptyl:
                    min_value=point
            
            p=0   
            x=0             
            while True:
                if   abs(points[x][0]-max_value[0])<rozptyl:
                        points.remove(points[x]) 
                        p=0
                x=x+1  
                if x>=len(points):
                    if p==1:
                        break
                    p=1   
                    x=0 
                if len(points)==0:
                    pomocny=1
                    break
    
            if math.sqrt((max_value[0] - min_value[0])**2 + (max_value[1] - min_value[1])**2)>rozptyl:
                bod1+=[max_value]
                bod2+=[min_value]
  
        bod11.append(bod1)
        bod21.append(bod2)
        dist=[]
        angles=[]
        #plt.imshow(text_color)
        for x in range(0,len(bod1)):
            #plt.plot([bod1[x][0],bod2[x][0]],[bod1[x][1],bod2[x][1]], linestyle='-', color='red')
            t=metody.intersectLines(maxl,maxp,bod1[x],bod2[x])
            r=math.sqrt((t[0] - maxl[0])**2 + (t[1] - maxl[1])**2)
            r=format(r,'.2f')
            dist.append(float(r))
            t=metody.angle(maxl,maxp,bod1[x],bod2[x])
            t=format(t,'.2f')
            angles.append(float(t))
         
        
        
        x = [
            { "filename": path,
              "incision_polyline": [[ maxl[0], maxl[1]],[maxp[0],maxp[1]]],
              "crossing_positions": dist,
              "crossing_angles": angles,
            }, 
          ]
        data.append(x)
        
        with open(name_out, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        '''   
        # vykresleni vysledku    
        plt.figure()
        plt.imshow(text_color)
        plt.figure()
        plt.plot([maxp[0],maxl[0]],[maxp[1],maxl[1]], linestyle='-', color='blue')
        plt.xlim(0, len(binary_image[0]))
        plt.ylim( len(binary_image),0)
        plt.imshow(text_color)
        for x in range(0,len(bod1)):
            plt.plot([bod1[x][0],bod2[x][0]],[bod1[x][1],bod2[x][1]], linestyle='-', color='red')
        '''    
    if visual_mode :
        for x in range(0,len(obr1)):            
            plt.figure()
            plt.imshow(obr1[x])
            plt.figure()
            plt.plot([maxp1[x][0],maxl1[x][0]],[maxp1[x][1],maxl1[x][1]], linestyle='-', color='blue')
        #plt.xlim(0, len(obr1[x][0]))
        #plt.ylim( len(obr1[x]),0)
            plt.imshow(obr1[x])
            for i in range(0,len(bod11[x])):
                #print(len(bod11[x][i]))
                plt.plot([bod11[x][i][0],bod21[x][i][0]],[bod11[x][i][1],bod21[x][i][1]], linestyle='-', color='red')
         
        plt.show()
    
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument("-v", "--visual_mode", action="store_true", help="Enable visual mode")
    parser.add_argument("image_files", nargs="+", help="Input image files")
    args = parser.parse_args()

    prog(args.output_file, args.visual_mode, *args.image_files)  #args.visual_mode,
