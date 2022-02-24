import numpy as np
from scipy import sparse


def mr(self, M, M1, M2):
     delta_E = np.ones((self.image_size,self.size_n))
     Wt = np.zeros((self.image_size,self.size_n))
     
     for i in range(self.image_size):
        ind = M[i].indices
        cou = ind.size
        for indi in range(cou):
           delta_E[i,ind[indi]] = sum(-M2[:,i]*M1[:,ind[indi]])       

     indices = []
     for m in range(self.image_size):   
        ind = M[m].indices
        indices.append(ind)

     for i in range(self.image_size):
        cou = indices[i].size   
        ATA = np.dot(M1[:,indices[i]].T, M1[:,indices[i]])  
        A = np.linalg.inv(ATA+self.lamda*np.eye(cou))        
        B = delta_E[i,indices[i]].T
        Wt[i,indices[i]] = -(np.dot(A,B)).T
#            print(i)
     Wt = sparse.csr_matrix(Wt)  
     return Wt 
