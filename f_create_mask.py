import math
import numpy as np
from scipy import sparse

def create_mask(args):  
        img_size = args.image_size
        d = img_size* img_size
        size_n = d+1
        brd = math.floor(args.receptive_field/2)
        masked = np.zeros((d,d))
        idx = 0
        inds1=[]
        for i in range(img_size):
            for j in range(img_size):
                inds2= []
                for m in range(i-brd,i+brd+1):
                    for n in range(j-brd,j+brd+1):
                        if m<0 or n<0:
                            continue
                        if m>=0 and m<img_size and n>=0 and n<img_size:
                             f = np.ravel_multi_index([m,n],(img_size,img_size),order='C')                                                               
                             inds1.append(f)
                    if len(inds1)!= 0:                        
                        inds2.append(inds1)
                    inds1=[]
                masked[idx,inds2]=1
                idx=idx+1  
          
        mask1 = np.ones((d,1))
        mask = np.zeros((d,size_n))
        mask[:,0:d] = masked
        mask[:,d:size_n] = mask1
        M = sparse.csr_matrix(mask) 
        return mask, M 
