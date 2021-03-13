# information theory metrics
import numpy as np


def ent(x, bins1, minval,range_x):
    # Calculates Shannon entropy of array x. 
    # Need to specify number of bins and range of x

    a=minval; b=a+range_x # 
    px=np.histogram(x, bins=bins1, density=True, range=(a,b)),
    rel=np.zeros(np.size(px[0][0]))
    p=px[0][0]#/np.size(x)
    binarea=px[0][1][1]-px[0][1][0]
    p=p*binarea
    for i in range(np.size(p)):
        if p[i]<1e-04:
            rel[i]=0
        else:
            rel[i]=p[i]*np.log2(1/p[i])

    ent=np.sum(rel)
    return ent

def mutualinfo(x,y,bins1,minval,range_x):
    # Calculates mutual information between x and y, 
    # Equal bins as bins1 and uses same range for both x and y
    #a=np.min(np.concatenate([x,y]));
    a1=minval; b=a1+range_x; #np.max(np.concatenate([x,y]))
    H1,a,b=np.histogram2d(x,y,bins=(bins1,bins1),density=True, range=((a1,b),(a1,b)))
    histshape=np.shape(H1)      #gives shape of hist/jpdf
    binarea=(a[2]-a[1])*(b[2]-b[1]) #bin area
    H=H1*binarea   #got the jpdf
    mpdfx=np.zeros(histshape[0]);   #mrpdfx - marginal pdf in x
    mpdfy=np.zeros(histshape[1]);   #mrpdfy - marginal pdf in y
    mpdfx=np.sum(H,axis=1)
    mpdfy=np.sum(H,axis=0)

    mut_ent=np.zeros(histshape)
    for i in range(histshape[0]):
        for j in range(histshape[1]):
            if H[i,j]<1e-04:
                mut_ent[i,j]=0;
            else:
                mut_ent[i,j]=H[i,j]*np.log2(H[i,j]/(mpdfx[i]*mpdfy[j]))
    ent=np.sum(mut_ent)
    return ent
