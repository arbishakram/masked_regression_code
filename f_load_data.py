import os
import glob
import numpy as np
import cv2
from normalize_images import normalize_img


def load_data(self, args):
    
             def get_img(img):
                if self.ch == 1:
                    image = cv2.imread(img,0)
                    image = normalize_img(image)   
                    img = np.zeros(self.image_size*self.ch)                
                    img[0:self.image_size] = np.reshape(image,[self.image_size])
                else:
                    image = cv2.imread(img)
                    image = normalize_img(image)   
                    img = np.zeros(self.image_size*self.ch)                
                    img[0:self.image_size] = np.reshape(image[:,:,0],[self.image_size])
                    img[self.image_size:self.image_size*2] = np.reshape(image[:,:,1],[self.image_size])
                    img[self.image_size*2:self.image_size*3] = np.reshape(image[:,:,2],[self.image_size])               
                return img
         
            
             if args.mode == 'test_inthewild':
                 in_path = args.inthewild_dataset_dir
                 imgA=glob.glob(os.path.join(in_path, '*.png'))
                 print('Number of images in '+str(args.mode)+':',len(imgA))
                 count = 0
                 for img in imgA:   
                    img_name = os.path.basename(img)
                    filename, ext = os.path.splitext(img_name)
                    self.names.append(filename)
                    img = get_img(img)
                    self.imagesA[count] = np.array(img)
                    count = count + 1 
                    
             else:
                 if args.mode == 'train':
                      in_path = args.train_dataset_dir
                 else:
                     in_path = args.test_dataset_dir
                
                 imgA=glob.glob(os.path.join(in_path, str(args.mode)+'A/*.png'))
                 print('Number of images in '+str(args.mode)+'A:',len(imgA))
                    
                 imgB=glob.glob(os.path.join(in_path, str(args.mode)+'B/*.png'))
                 print('Number of images in '+str(args.mode)+'B:',len(imgB))
                
                 for img in imgA:   
                    img_name = os.path.basename(img)
                    filename, ext = os.path.splitext(img_name)
                    num = filename.split('_')[1]
                    self.names.append(filename)
                    img = get_img(img)
                    self.imagesA[int(num)-1] = np.array(img)
                 
                 for img in imgB:
                    img_name = os.path.basename(img)
                    filename, ext = os.path.splitext(img_name)
                    num = filename.split('_')[1]
                    img = get_img(img)
                    self.imagesB[int(num)-1] = np.array(img)
                    
             return self.imagesA, self.imagesB
         
            
            
def load_x_t(self, p,ch,imagesA,imagesB):           
            if ch==0:        
                x = imagesA[p,0:self.image_size] 
                t = imagesB[p,0:self.image_size]
            if ch==1:
                x = imagesA[p,self.image_size:self.image_size*2] 
                t = imagesB[p,self.image_size:self.image_size*2]
                
            if ch==2:
                x = imagesA[p,self.image_size*2:self.image_size*3] 
                t = imagesB[p,self.image_size*2:self.image_size*3]
            xt = np.ones(self.size_n)
            xt[0:self.image_size] = x
            return xt,t
        
            
        
def load_x(self, p, ch):           
            if ch==0:        
                x = self.imagesA[p,0:self.image_size] 
            if ch==1:
                x = self.imagesA[p,self.image_size:self.image_size*2]                    
            if ch==2:
                x = self.imagesA[p,self.image_size*2:self.image_size*3] 
            xt = np.ones(self.size_n)   
            xt[0:self.image_size] = x              
            return xt

                

def create_design_response_matrices(self, M,ch, imagesA, imagesB):
        M1 = np.ones((self.n, self.size_n))
        M2 = np.zeros((self.n, self.image_size))
        r=0
        for p in range(self.n):  
                            x, t = load_x_t(self, p,ch,imagesA,imagesB) 
                            if len(x) or len(t) != 0:
                                x = x.T                 
                                tt = np.reshape(t,[1,self.image_size])  
                                M1[r,:] = x
                                M2[r,:] = tt
                                r = r+1
            
        return M1, M2
    