# calculates z from scoord

import numpy as np

def z4ms(sig,h,zeta):
    # gets a single array z
    csig1= (1 - np.cosh(sig))/(np.cosh(1) - 1)
    Csig2=(np.exp(csig1)-1)/(1-np.exp(-1))
    
    S = (3*sig + h*Csig2)/(3+h)
    z=zeta+(zeta+h)*S
    return z

def zgrid4ms(sig,h,zeta):
    # gets a z-grid out of zeta, h, and scoord
    z=np.zeros([np.shape(zeta)[0],len(sig),np.shape(zeta)[1],np.shape(zeta)[2]]);
    csig1=np.zeros(np.shape(z))
    csig2=np.zeros(np.shape(z))
    sig_grid=np.zeros(np.shape(z))
    h1=np.zeros(np.shape(z))
    zeta_grid=np.zeros(np.shape(z))
      
    for i in range(np.shape(zeta)[0]):
        for j in range(len(sig)):
            h1[i,j,:,:]=h
            
    for k in range(len(sig)):
        csig1[:,k,:,:]=(1 - np.cosh(sig[k]))/(np.cosh(1) - 1)
        sig_grid[:,k,:,:]=sig[k]
        zeta_grid[:,k,:,:]=zeta
    
    Csig2=(np.exp(csig1)-1)/(1-np.exp(-1))
    
    z=(3*sig_grid + h1*Csig2)/(3+h1)
    z=zeta_grid+(zeta_grid+h1)*z
    return z 
