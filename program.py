import os
import sys
import numpy as np
from skimage import io
import cv2

from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

def getCorners(img,N,harris):
    corners = cv2.goodFeaturesToTrack(img,500,0.01,5,blockSize=5,useHarrisDetector=harris)
    corners = corners.astype(int)
    #print(corners[0])
    #corners = np.int(corners)
    cleaned=[]
    for i in range(corners.shape[0]):#filter those too close to the edge away
        if(corners[i,0,0]-N<0 or corners[i,0,1]-N<0 or corners[i,0,0]+N > img.shape[1] or corners[i,0,1]+N > img.shape[0]):
            continue
        else:
            cleaned.append(corners[i,0,:])
    return np.array(cleaned)

def similarityPatches(p1,p2):# sum of squared intensity differences.
    rows = p1.shape[0]
    cols = p1.shape[1]
    val = 0
    for x in range(0, rows):
        for y in range(0, cols):
            val += (int(p1[x,y]) - int(p2[x,y])) **2
    return val

def matchUp(im1,im2,ps1,ps2,N,sim=0.7):
    cons=[]
    dissims=[]
    for i in range(len(ps1)):
        sims=[]
        for j in range(len(ps2)):
            c1=ps1[i,:]
            c2=ps2[j,:]
            p1=im1[c1[1]-N:c1[1]+N+1, c1[0]-N:c1[0]+N+1]
            p2=im2[c2[1]-N:c2[1]+N+1, c2[0]-N:c2[0]+N+1]
            #print(c1,c2)
            #print(ps1,ps2)
            sims.append(similarityPatches(p1,p2))
        res=np.argsort(sims)[0:2]
        if(sims[res[1]]==0):
            continue
        dissim=sims[res[0]]/sims[res[1]]
        if(dissim>sim):#dissimilarity
            continue
        else: #left-right check
            simsback=[]
            for ii in range(len(ps1)):
                c1=ps1[ii,:]
                c2=ps2[res[0],:]
                p1=im1[c1[1]-N:c1[1]+N+1, c1[0]-N:c1[0]+N+1]
                p2=im2[c2[1]-N:c2[1]+N+1, c2[0]-N:c2[0]+N+1]
                simsback.append(similarityPatches(p1,p2))
            icheck=np.argsort(simsback)[0]
            if(i==icheck):
                cons.append((i,res[0]))
                dissims.append(sims[res[0]])
    return cons,dissims


def showMatching(img1,img2,ps1,ps2,cons):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1,"gray")
    ax1.plot(ps1[:,0],ps1[:,1],'y.')

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2,"gray")
    ax2.plot(ps2[:,0],ps2[:,1],'y.')

    for i in range(len(cons)):
        (p1,p2)=cons[i]
        cp = ConnectionPatch(xyA=ps1[p1,:], xyB=ps2[p2,:], coordsA='data', coordsB='data',
                      axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(cp)


def statistics(f1,f2,cons,dissims):#mean,std feats 1 and 2, cons
    print("Features 1: {}".format(f1.shape[0]))
    print("Features 2: {}".format(f2.shape[0]))
    print("Connection Count: {}".format(len(dissims)))
    print("Mean: {}".format(np.mean(dissims)))
    print("Std: {}".format(np.std(dissims)))

def findBestParams(img1,img2):

    for i in range(2,7):
        ps1=getCorners(img1,i,True)
        ps2=getCorners(img2,i,True)
        cons,dissims = matchUp(img1,img2,ps1,ps2,i)
        print("N radius {} gives: {}".format(i,np.std(dissims)/i))


def loadImages():
    imageName1 = "Img001_diffuse.tif"
    imageName2 = "Img002_diffuse.tif"
    imageName3 = "Img009_diffuse.tif"
    im_path1 = os.path.join(sys.path[0], imageName1)
    im_path2 = os.path.join(sys.path[0], imageName2)
    im_path3 = os.path.join(sys.path[0], imageName3)
    im1=cv2.imread(im_path1,0)
    im2=cv2.imread(im_path2,0)
    im3=cv2.imread(im_path3,0)
    return im1,im2,im3

def performMatching(im1,im2,K,sim):
    ps1=getCorners(im1,K,True)
    ps2=getCorners(im2,K,True)
    cons, dissims = matchUp(im1,im2,ps1,ps2,K,sim)
    showMatching(im1,im2,ps1,ps2,cons)
    statistics(ps1,ps2,cons,dissims)
    plt.show()

im1,im2,im3=loadImages()
performMatching(im1,im2,5,0.6)
#corners = cv2.goodFeaturesToTrack(im1,500,0.01,5,blockSize=5,useHarrisDetector=True)
#findBestParams(im1,im2)




