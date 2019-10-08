# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:38:17 2019

@author: Raul Alfredo de Sousa Silva
Features extraction

"""
# Imports (used libraries)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morpho
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import AxesGrid


############################################################################
# Basic function from ancient TPs of IMA coures
############################################################################
def Get_values_without_error(im,XX,YY):
    """ retouren une image de la taille de XX et YY 
     qui vaut im[XX,YY] mais en faisant attention a ce que XX et YY ne debordent
     pas """
    sh=XX.shape
    defaultval=0;
    if len(im.shape)>2: #color image !
        defaultval=np.asarray([0,0,0])
        sh=[*sh,im.shape[2]]
    imout=np.zeros(sh)
    (ty,tx)=XX.shape[0:2]
    for k in range(ty):
        for l in range(tx):
            posx=int(XX[k,l]-0.5)
            posy=int(YY[k,l]-0.5)
            if posx<0 or posx>=im.shape[1] or posy<0 or posy>=im.shape[0]:
                valtmp=defaultval
            else:
                valtmp=im[posy,posx]
            imout[k,l]=valtmp
     
    return imout

def rotation(im,theta,alpha=1.0,x0=None,y0=None,ech=0,clip=True):
    """
   %
%Effectue la transformation geometrique d'une image par
%une rotation + homothetie 
%
% x' = alpha*cos(theta)*(x-x0) - alpha*sin(theta)*(y-y0) + x0
% y' = alpha*sin(theta)*(x-x0) + alpha*cos(theta)*(y-y0) + y0 
%
% theta : angle de rotation en degres
% alpha : facteur d'homothetie (defaut=1)
% x0, y0 : centre de la rotation (defaut=centre de l'image)
% ech : plus proche voisin (defaut=0) ou bilineaire (1)
% clip : format de l'image originale (defaut=True), image complete (False)
% 
 
    """
    dy=im.shape[0]
    dx=im.shape[1]
     
    if x0 is None:
        x0=dx/2.0
    if y0 is None:
        y0=dy/2.0
    v0=np.asarray([x0,y0]).reshape((2,1))
    theta=theta/180*np.pi
    ct=alpha*np.cos(theta)
    st=alpha*np.sin(theta)
    matdirect=np.asarray([[ct,-st],[st,ct]])
    if clip==False:
        #ON CALCULE exactement la transformee des positions de l'image
        # on cree un tableau des quatre points extremes
        tabextreme=np.asarray([[0,0,dx,dx],[0,dy,0,dy]])
        tabextreme_trans= matdirect@(tabextreme-v0)+v0
        xmin=np.floor(tabextreme_trans[0].min())
        xmax=np.ceil(tabextreme_trans[0].max())
        ymin=np.floor(tabextreme_trans[1].min())
        ymax=np.ceil(tabextreme_trans[1].max())
         
    else:
        xmin=0
        xmax=dx
        ymin=0
        ymax=dy
    if len(im.shape)>2:
        shout=(int(ymax-ymin),int(xmax-xmin),im.shape[2]) # image couleur
    else:
        shout=(int(ymax-ymin),int(xmax-xmin))
    dyout=shout[0]
    dxout=shout[1]
    eps=0.0001
    Xout=np.arange(xmin+0.5,xmax-0.5+eps)
    Xout=np.ones((dyout,1))@Xout.reshape((1,-1)) 
     
    Yout=np.arange(ymin+0.5,ymax-0.5+eps)
    Yout=Yout.reshape((-1,1))@np.ones((1,dxout))
     
    XY=np.concatenate((Xout.reshape((1,-1)),Yout.reshape((1,-1))),axis=0)
    XY=np.linalg.inv(matdirect)@(XY-v0)+v0
    Xout=XY[0,:].reshape(shout)
    Yout=XY[1,:].reshape(shout)
    if ech==0: # plus proche voisin
        out=Get_values_without_error(im,Xout,Yout)
    else:  #bilineaire 
        assert ech == 1 , "Vous avez choisi un echantillonnage inconnu"
        Y0=np.floor(Yout-0.5)+0.5 # on va au entier+0.5 inferieur
        X0=np.floor(Xout-0.5)+0.5
        Y1=np.ceil(Yout-0.5)+0.5
        X1=np.ceil(Xout-0.5)+0.5
        PoidsX=Xout-X0
        PoidsY=Yout-Y0
        PoidsX[X0==X1]=1 #points entiers
        PoidsY[Y0==Y1]=1 #points entiers
        I00=Get_values_without_error(im,X0,Y0)
        I01=Get_values_without_error(im,X0,Y1)
        I10=Get_values_without_error(im,X1,Y0)
        I11=Get_values_without_error(im,X1,Y1)
        out=I00*(1.0-PoidsX)*(1.0-PoidsY)+I01*(1-PoidsX)*PoidsY+I10*PoidsX*(1-PoidsY)+I11*PoidsX*PoidsY
    return out

def strel(forme,taille,angle=45):
    """renvoie un element structurant de forme  
     'diamond'  boule de la norme 1 fermee de rayon taille
     'disk'     boule de la norme 2 fermee de rayon taille
     'square'   carre de cote taille (il vaut mieux utiliser taille=impair)
     'line'     segment de langueur taille et d'orientation angle (entre 0 et 180 en degres)
      (Cette fonction n'est pas standard dans python)
    """
 
    if forme == 'diamond':
        return morpho.selem.diamond(taille)
    if forme == 'disk':
        return morpho.selem.disk(taille)
    if forme == 'square':
        return morpho.selem.square(taille)
    if forme == 'line':
        angle=int(-np.round(angle))
        angle=angle%180
        angle=np.float32(angle)/180.0*np.pi
        x=int(np.round(np.cos(angle)*taille))
        y=int(np.round(np.sin(angle)*taille))
        if x**2+y**2 == 0:
            if abs(np.cos(angle))>abs(np.sin(angle)):
                x=int(np.sign(np.cos(angle)))
                y=0
            else:
                y=int(np.sign(np.sin(angle)))
                x=0
        rr,cc=morpho.selem.draw.line(0,0,y,x)
        rr=rr-rr.min()
        cc=cc-cc.min()
        img=np.zeros((rr.max()+1,cc.max()+1) )
        img[rr,cc]=1
        return img
    raise RuntimeError('Erreur dans fonction strel: forme incomprise')
    
def extract(image,seg,img_id):
###########################################################################
    # Color channels - 6 space colors
###########################################################################
    # Creating channels in the 6 space colors
    
    #RGB
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    #Normalized RGB
    image_n = image.astype(np.uint16)
    r = image_n[:,:,0]/(image_n[:,:,0]+image_n[:,:,1]+image_n[:,:,2]+0.00001)
    g = image_n[:,:,1]/(image_n[:,:,0]+image_n[:,:,1]+image_n[:,:,2]+0.00001)
    b = image_n[:,:,2]/(image_n[:,:,0]+image_n[:,:,1]+image_n[:,:,2]+0.00001)
    
    #HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = image_hsv[:,:,0]
    s = image_hsv[:,:,1]
    v = image_hsv[:,:,2]
    
    #I1/2/3
    I1 = (R+G+B)/3
    I2 = (R+B)/2
    I3 = (2*G-R-B)/4
    #image_I = np.transpose(np.array([I1,I2,I3]))
    
    #l1/2/3
    l1 = (R-G)**2/((R-G)**2+(R-B)**2+(G-B)**2+.000001)
    l2 = (R-B)**2/((R-G)**2+(R-B)**2+(G-B)**2+.000001)
    l3 = (G-B)**2/((R-G)**2+(R-B)**2+(G-B)**2+.000001)
    #image_l = np.transpose(np.array([l1,l2,l3]))
    
    # CIE l*a*b*
    image_luv = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
    l = image_luv[:,:,0]
    u = image_luv[:,:,1]
    v2 = image_luv[:,:,2]
    
    channels = [G,B,r,g,b,h,s,v,I1,I2,I3,l1,l2,l3,l,u,v2]
    colors = np.expand_dims(R, axis=2)
    for element in channels:
        colors = np.append(colors, np.expand_dims(element, axis=2),axis=2)
    del channels,R,G,B,r,g,b,h,s,v,I1,I2,I3,l1,l2,l3,l,u,v2
    del image_hsv,image_luv,image_n
############################################################################    
    # Shape Features - 13
############################################################################    
    [m,n] = np.shape(seg)
    ind = np.where(seg==1)
    row = np.expand_dims(ind[0], axis=1)
    col = np.expand_dims(ind[1], axis=1)
    
    # First order geometric moments
    m10 = sum(row)
    m01 = sum(col)
    m00 = sum(sum(seg))
    r0 = m10/m00
    c0 = m01/m00
    
    # Second order central moments
    mu02 = sum((col-c0)**2)
    mu20 = sum((row-r0)**2)
    mu11 = sum((col-c0)*(row-r0))
    m10c = sum(row*colors[ind])
    m01c = sum(col*colors[ind])
    m00c = sum(colors[ind])
    mu02c = sum((col-c0)**2*image[ind])
    mu20c = sum((row-r0)**2*image[ind])
    mu11c = sum((col-c0)*(row-r0)*image[ind])
    
    # Perimeter
    perimeter = 0
    for k in range(0,len(ind[0])):
        i = ind[0][k]
        j = ind[1][k]
        zeroc = max(0,j-1)
        infic = min(n-1,j+1)
        zeror = max(0,i-1)
        infir = min(m-1,i+1)
        if (seg[i,infic]+seg[i,zeroc]+seg[infir,j]+seg[zeror,j])<4:
            perimeter+=1
    
    
    # Aspect ratio           
    L1 = (8*(mu02+mu20 + ((mu02-mu20)**2 + 4*mu11)**(1/2) ))**(1/2)
    L2 = (8*(mu02+mu20 - ((mu02-mu20)**2 + 4*mu11)**(1/2) ))**(1/2)
    Ar = L1/L2
    epsilon = ((mu02-mu20)**2 + 4*mu11)/((mu02-mu20)**2)
    theta = 0.5*np.tan(2*mu11/(mu20-mu02))
    
    mu20 = mu20
    mu02 = mu02
    mu11 = mu11
    # Assimetry 1 and 2
        # Rotation around the centroid of the lesion
    boolean_x = rotation(seg,180*theta[0]/np.pi,x0=round(c0[0]),y0=round(r0[0]))
    boolean_x = boolean_x.astype(np.uint8) # Cast to integer
        # number of pixels that belongs to both images (original an rotated)
    diffx = sum(sum(boolean_x*seg))
    
    boolean_y = rotation(seg,180*theta[0]/np.pi-90,x0=round(c0[0]),y0=round(r0[0]))
    boolean_y = boolean_y.astype(np.uint8) # Cast to integer
        # number of pixels that belongs to both images (original an rotated)
    diffy = sum(sum(boolean_y*seg))
    Ax = m00-diffx
    Ay = m00-diffy
    A1 = min(Ax,Ay)/m00
    A2 = (Ax+Ay)/m00
    
    #Equivalent diameter parametrised by the axis L1 ond (or) L2
    eqd = round(2*np.sqrt(m00/np.pi))
    L1 = L1
    L2 = L2
    # Shape features
    shapes = [m00,eqd,perimeter,Ar[0],epsilon[0],theta[0],A1,A2,L1[0],L2[0],mu20[0],mu02[0],mu11[0]]
    
    # Grouping all features
    metrics = [img_id]
    for element in shapes:
        metrics.append(element)
        
    del shapes
#############################################################################
    # Color features - 348
#############################################################################
    se=strel('disk',round(0.05*eqd/2))
    se2=strel('disk',round(0.1*eqd/2))
    # dilatation
    ignore=morpho.dilation(seg,se)
    consider = morpho.dilation(seg,se2)
    inner = consider-ignore
    # erosion
    ignore=morpho.erosion(seg,se)
    consider = morpho.erosion(seg,se2)
    outer = ignore-consider
    
    seg_ex = np.expand_dims(seg, axis=2)
    inner_ex = np.expand_dims(inner, axis=2)
    outer_ex = np.expand_dims(outer, axis=2)
    
    col_seg = (seg_ex*colors)
    col_inner = (inner_ex*colors)
    col_outer = (outer_ex*colors)
    
    # 324 features of mean and standard deviation
    for i in range(0,colors.shape[2]):
        metrics.append(np.mean(col_seg[:,:,i]))
        metrics.append(np.std(col_seg[:,:,i]))
        metrics.append(np.mean(col_inner[:,:,i]))
        metrics.append(np.std(col_inner[:,:,i]))
        metrics.append(np.mean(col_outer[:,:,i]))
        metrics.append(np.std(col_outer[:,:,i]))
        metrics.append(np.mean(col_outer[:,:,i])/np.mean(col_inner[:,:,i]))
        metrics.append(np.std(col_outer[:,:,i])/np.std(col_inner[:,:,i]))
        metrics.append(np.mean(col_outer[:,:,i])/np.mean(col_seg[:,:,i]))
        metrics.append(np.std(col_outer[:,:,i])/np.std(col_seg[:,:,i]))
        metrics.append(np.mean(col_inner[:,:,i])/np.mean(col_seg[:,:,i]))
        metrics.append(np.std(col_inner[:,:,i])/np.std(col_seg[:,:,i]))
        metrics.append(np.mean(col_outer[:,:,i])-np.mean(col_inner[:,:,i]))
        metrics.append(np.std(col_outer[:,:,i])-np.std(col_inner[:,:,i]))
        metrics.append(np.mean(col_outer[:,:,i])-np.mean(col_seg[:,:,i]))
        metrics.append(np.std(col_outer[:,:,i])-np.std(col_seg[:,:,i]))
        metrics.append(np.mean(col_inner[:,:,i])-np.mean(col_seg[:,:,i]))
        metrics.append(np.std(col_inner[:,:,i])-np.std(col_seg[:,:,i]))
    
    # 6 features of color asymmetry
    thetacol = 0.5*np.tan(2*mu11c/(mu20c-mu02c))
        # Rotation around the centroid of the lesion
    for i in range(0,3):
        thetac = thetacol[i]
        boolean_x = rotation(seg,180*thetac/np.pi,x0=round(m01c[i]/m00),y0=round(m10c[i]/m00))
        boolean_x = boolean_x.astype(np.uint8) # Cast to integer
        diffx = sum(sum(boolean_x*seg))
        boolean_y = rotation(seg,180*thetac/np.pi-90,x0=round(m01c[i]/m00),y0=round(m10c[i]/m00))
        boolean_y = boolean_y.astype(np.uint8) # Cast to integer
        diffy = sum(sum(boolean_y*seg))
        Ax = m00-diffx
        Ay = m00-diffy
        A1 = min(Ax,Ay)/m00
        A2 = (Ax+Ay)/m00
        metrics.append(A1)
        metrics.append(A2)
    
    # 18 features of centroidal distances
    r0c = m10c/m00c
    c0c = m01c/m00c
    dist = (r0-r0c)**2+(c0-c0c)**2
    # Normalizing
    dist = dist/max(dist)
    for i in range(0,len(dist)):
        metrics.append(dist[i])
    del colors
############################################################################    
    # Texture features - 72
############################################################################
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #imshow(gray)
    comp = np.array([seg,inner,outer])
    ind = np.where(comp == 1)
    # Requantized image
    ncolor = 64
    img_c = np.round(np.array([ncolor-1])*gray/255).astype(np.uint8)
    # Co-occurence matrix for 0°(0), 45°(1), 90°(2), 135°(3)
    C = np.zeros([64,64,3,4])
    # Size of the masks (to normalization)
    size = np.array([[[sum(sum(seg)),sum(sum(inner)),sum(sum(outer))]]])
    for l in range(0,len(ind[0])):
        k = ind[0][l]
        i = ind[1][l]
        j = ind[2][l]
        maxi = min(n-1,j+1)
        minij = max(0,j-1)
        minii = max(0,i-1)
        C[img_c[i,j],img_c[i,maxi],k,0] +=1
        C[img_c[i,j],img_c[minii,maxi],k,1] +=1
        C[img_c[i,j],img_c[minii,j],k,2] +=1
        C[img_c[i,j],img_c[minii,minij],k,3] +=1
    
    Cmean = np.mean(C,axis=3)/size
    
    # Maximum probability
    mp = np.max(Cmean,axis=(0,1))
    # Energy
    E = sum(sum(Cmean**2))
    # Entropy
    S = -sum(sum(Cmean*np.log(Cmean+0.00001)))
    # Auxiliary matrix with |i-j| to each (i,j)
    pos = np.linspace(0,ncolor-1,ncolor)
    m1 = np.matlib.repmat(pos,ncolor,1)
    o = np.ones([ncolor,ncolor])
    mpos = abs(m1 - np.transpose(pos*o))
    mpos = np.expand_dims(mpos, axis=2)
    m = np.expand_dims(m1, axis=2)
    # Dissimilarity
    D = sum(sum(mpos*Cmean))
    # Contrast
    C = sum(sum(mpos**2*Cmean))
    # Inverse difference
    ID = sum(sum(Cmean/(1+mpos)))
    # Inverse difference moment
    IDM = sum(sum(Cmean/(1+mpos**2)))
    # Auxiliary variables
    muj = sum(sum(m*Cmean))
    mui = sum(sum(np.transpose(m,(1,0,2))*Cmean))
    sigmaj = sum(sum((m-muj)**2*Cmean))
    sigmai = sum(sum((np.transpose(m,(1,0,2))-mui)**2*Cmean))
    # Correlation
    COR = sum(sum((np.transpose(m,(1,0,2))-mui)*(m-muj)*Cmean/(sigmai*sigmaj)))
    # Texture features for seg, inner and outer
    t = np.array([mp,E,S,D,C,ID,IDM,COR])
    del mp,E,S,D,C,ID,IDM,COR,mpos,o,m,pos,m1,Cmean
    # Expanding texture features: same operations made for color (- and /)
    new1 = np.transpose(np.array([t[:,2]-t[:,1],t[:,2]-t[:,0],t[:,1]-t[:,0]]))
    new2 = np.transpose(np.array([t[:,2]/t[:,1],t[:,2]/t[:,0],t[:,1]/t[:,0]]))
    
    for i in range(0,8):
        for j in range(3):
            metrics.append(t[i,j])
            metrics.append(new1[i,j])
            metrics.append(new2[i,j])
    return metrics
