import numpy as np
import cv2
from scipy import sparse
from f_create_mask import create_mask
from f_load_data import *
from f_mr import mr
import json

class mr_solver:
     """Solver for training and testing MR."""
     
     def __init__(self,args):
        self.size = args.image_size
        self.image_size = self.size * self.size 
        self.size_n = self.image_size+1
        self.n = args.total_images
        self.r = args.receptive_field
        self.ch = args.input_ch
        self.lamda = args.lamda
        self.weight_dir = args.weights_dir
        self.train_dataset_dir = args.train_dataset_dir
        self.test_dataset_dir = args.test_dataset_dir
        self.inthewild_dataset_dir = args.inthewild_dataset_dir
        self.names = []
        self.imagesA = np.zeros((self.n,self.image_size*self.ch))
        self.imagesB = np.zeros((self.n,self.image_size*self.ch))

     def train(self, args):       
         path = str(self.weight_dir)+'arguments_history.txt'
         with open(path, 'w') as f:
                 json.dump(args.__dict__, f, indent=2)           
         print("loading input and target images...")
         imagesA, imagesB = load_data(self, args)   
         print("construct a mask with "+str(self.r)+"x"+str(self.r)+" receptive field.....")
         mask, M = create_mask(args)
         
         if self.ch == 3:
             print("*"*40)             
             print("learn MR weights for each channel seperately..")         
             print("*"*40)
         
         for ch in range(self.ch):
             print("form design and repsonse matrices.... ")
             M1, M2 = create_design_response_matrices(self, M, ch, imagesA, imagesB)
             print("learn MR weights....")
             W = mr(self, M, M1, M2)
             sparse.save_npz(str(args.weights_dir)+'mr_weights_dec-10-2019_'+str(ch+1)+'.npz', W)
             print("saved weight matrix "+str(ch+1)+"...")
             print("*"*40)
         print("Done")
             
    
     def test(self, args):
         if args.mode == 'test':
             print("loading input and target images...")
             imagesA, imagesB = load_data(self, args)
             for p in range(self.n):
                 tn = xn = np.zeros((self.size,self.size,self.ch))   
                 ynt = np.zeros((self.size,self.size,self.ch))
                 for ch in range(self.ch):                                                    
                        x, t = load_x_t(self, p,ch, imagesA,imagesB) 
                        Wt = sparse.load_npz(str(args.weights_dir)+'mr_weights_dec-10-2019_'+str(ch+1)+'.npz') 
                        Wt = Wt.todense()
                        if len(x)  != 0:  
                           yn = np.dot(Wt,x) 
                           xnt = x[0:self.image_size]
                           xn[:,:,ch] = np.reshape(xnt,(self.size,self.size))
                           ynt[:,:,ch] = np.reshape(yn,(self.size,self.size))
                           tn[:,:,ch] = np.reshape(t,(self.size,self.size))
                 ynt = cv2.normalize(ynt, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                 xn = cv2.normalize(xn, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)   
                 tn = cv2.normalize(tn, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)   
                 ftest = np.concatenate((xn,ynt,tn), axis=1) 
                 cv2.imwrite(str(args.test_results_dir)+str(self.names[p])+'_'+str(args.mode)+'.png',ftest)
                 print('Saved input, output and target images into {}'.format(args.test_results_dir))
             
         else:
             print("loading in the wild images...")
             load_data(self, args)
             for p in range(self.n):
                 xn = np.zeros((self.size,self.size,self.ch))   
                 ynt = np.zeros((self.size,self.size,self.ch))
                 for ch in range(self.ch):                                                    
                        x = load_x(self, p,ch) 
                        Wt = sparse.load_npz(str(args.weights_dir)+'mr_weights_dec-10-2019_'+str(ch+1)+'.npz') 
                        Wt = Wt.todense()
                        if len(x)  != 0:  
                           yn = np.dot(Wt,x)                                   
                           xnt = x[0:self.image_size]
                           xn[:,:,ch] = np.reshape(xnt,(self.size,self.size))
                           ynt[:,:,ch] = np.reshape(yn,(self.size,self.size))                          
                 ynt = cv2.normalize(ynt, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                 xn = cv2.normalize(xn, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)                  
                 ftest = np.concatenate((xn,ynt), axis=1) 
                 cv2.imwrite(str(args.inthewild_results_dir)+str(self.names[p])+'_'+str(args.mode)+'.png',ynt)
                 print('Saved input and output images into {}'.format(args.inthewild_results_dir))
