#!python3
#coding=utf-8
## illumination_correction

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from pylab import *

config = {
    "sigma_list": [15, 80, 250],
    "G"         : 5.0,
    "b"         : 25.0,
    "alpha"     : 125.0,
    "beta"      : 46.0,
    "low_clip"  : 0.01,
    "high_clip" : 0.99
}

class IlluminationCorrector:
    def __init__(self,img,conf = config):
        self.img = img # cv2 format
    def ID(self):       # identity transform
        return self.img
    
    def HE(self):       # histogram equalization
        (b,g,r) = cv2.split(self.img)#图像通道分解
        Eb = cv2.equalizeHist(b)
        Eg = cv2.equalizeHist(g)
        Er = cv2.equalizeHist(r)
        BGRresult = cv2.merge((Eb,Eg,Er))
        return BGRresult
    
    def AHE(self):       # adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        b = clahe.apply(self.img[:, :, 0])
        g = clahe.apply(self.img[:, :, 1])
        r = clahe.apply(self.img[:, :, 2])
        BGRresult = cv2.merge((b,g,r))
        return BGRresult

    def LT(self):
        log_img = np.log(1+np.float32(self.img)) 
        cv2.normalize(log_img, log_img, 0, 255, cv2.NORM_MINMAX)
        log_img = cv2.convertScaleAbs(log_img)
        return log_img
    
    def GIC(self, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.img, table)
    
    def replaceZeroes(self, data):
        min_nonzero = min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data

    def singleChannelSSR(self, src_img, size = 3):
        L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
        img = replaceZeroes(src_img)
        L_blur = replaceZeroes(L_blur)

        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
        log_R = cv2.subtract(dst_Img, dst_IxL)

        dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
        log_uint8 = cv2.convertScaleAbs(dst_R)
        return log_uint8
    
    def SSR(self,size=3):  
        b,g,r = cv2.split(self.img)
        b = singleChannelSSR(b,size)
        g = singleChannelSSR(g,size)
        r = singleChannelSSR(r,size)
        return cv2.merge([b, g, r])
        
    def singleScaleRetinex(self, img, sigma):
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        return retinex
        
    def multiScaleRetinex(self,img, sigma_list):
        retinex = np.zeros_like(img)
        for sigma in sigma_list:
            retinex += singleScaleRetinex(img, sigma)
        retinex = retinex / len(sigma_list)
        return retinex

    def colorRestoration(self, img, alpha, beta):
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
        return color_restoration

    def simplestColorBalance(self, img, low_clip, high_clip):    
        total = img.shape[0] * img.shape[1]
        for i in range(img.shape[2]):
            unique, counts = np.unique(img[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):            
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
                    
            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
        return img
    


    def MSRCR(self, sigma_list = config['sigma_list'], G = config['G'], b = config['b'], alpha = config['alpha'], beta = config['beta'], low_clip = config['low_clip'], high_clip = config['high_clip']):
        img = np.float64(self.img) + 1.0
        img_retinex = multiScaleRetinex(img, sigma_list)    
        img_color = colorRestoration(img, alpha, beta)    
        img_msrcr = G * (img_retinex * img_color + b)

        for i in range(img_msrcr.shape[2]):
            img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                                 (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                                 255
        
        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)       
        return img_msrcr

    def automatedMSRCR(self, sigma_list = config['sigma_list']):
        img = np.float64(self.img) + 1.0
        img_retinex = multiScaleRetinex(img, sigma_list)
        for i in range(img_retinex.shape[2]):
            unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
            for u, c in zip(unique, count):
                if u == 0:
                    zero_count = c
                    break
                
            low_val = unique[0] / 100.0
            high_val = unique[-1] / 100.0
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break
                
            img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
            
            img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                                   (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                                   * 255
        img_retinex = np.uint8(img_retinex)
        return img_retinex

    def MSRCP(self, sigma_list = config['sigma_list'], low_clip = config['low_clip'], high_clip = config['high_clip']):
        img = np.float64(self.img) + 1.0
        intensity = np.sum(img, axis=2) / img.shape[2]    
        retinex = multiScaleRetinex(intensity, sigma_list)
        intensity = np.expand_dims(intensity, 2)
        retinex = np.expand_dims(retinex, 2)
        intensity1 = simplestColorBalance(retinex, low_clip, high_clip)
        intensity1 = (intensity1 - np.min(intensity1)) / \
                     (np.max(intensity1) - np.min(intensity1)) * \
                     255.0 + 1.0
        img_msrcp = np.zeros_like(img)
        for y in range(img_msrcp.shape[0]):
            for x in range(img_msrcp.shape[1]):
                B = np.max(img[y, x])
                A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
                img_msrcp[y, x, 0] = A * img[y, x, 0]
                img_msrcp[y, x, 1] = A * img[y, x, 1]
                img_msrcp[y, x, 2] = A * img[y, x, 2]
        img_msrcp = np.uint8(img_msrcp - 1.0)
        return img_msrcp


# Histogram equalization
def HE(img):
    (b,g,r) = cv2.split(img)#图像通道分解
    Eb = cv2.equalizeHist(b)
    Eg = cv2.equalizeHist(g)
    Er = cv2.equalizeHist(r)
    BGRresult = cv2.merge((Eb,Eg,Er))
    return BGRresult
# Adaptive Histogram equalization
def AHE(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    b = clahe.apply(img[:, :, 0])
    g = clahe.apply(img[:, :, 1])
    r = clahe.apply(img[:, :, 2])
    BGRresult = cv2.merge((b,g,r))
    return BGRresult

# logarithmic transformation
def LT(img):
    log_img = np.log(1+np.float32(img)) 
    cv2.normalize(log_img, log_img, 0, 255, cv2.NORM_MINMAX)
    log_img = cv2.convertScaleAbs(log_img)
    return log_img
    
# gamma intensity correction
def GIC(image, gamma=1.0):

	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

# has problem
def LoG(img):
    #kernel = np.array(([0,-1,0],[-1,-5,-1],[0,-1,0]),dtype=np.float32)
    #dst = cv2.filter2D(np.float32(img),-1,kernel)
    blur = cv2.GaussianBlur(img,(3,3),0)
    laplacian = cv2.Laplacian(blur,cv2.CV_64F)
    laplacian1 = laplacian/laplacian.max()
    return laplacian1
    
def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def singleChannelSSR(src_img, size = 3):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def SSR(img,size=3):
    b,g,r = cv2.split(img)
    b = singleChannelSSR(b,size)
    g = singleChannelSSR(g,size)
    r = singleChannelSSR(r,size)
    return cv2.merge([b, g, r])


def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex

def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration

def simplestColorBalance(img, low_clip, high_clip):    
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):            
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
                
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img    

def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, sigma_list)    
    img_color = colorRestoration(img, alpha, beta)    
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255
    
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)       
    return img_msrcr

def automatedMSRCR(img, sigma_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, sigma_list)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex

def MSRCP(img, sigma_list, low_clip, high_clip):
    img = np.float64(img) + 1.0
    intensity = np.sum(img, axis=2) / img.shape[2]    
    retinex = multiScaleRetinex(intensity, sigma_list)
    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)
    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0
    img_msrcp = np.zeros_like(img)
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]
    img_msrcp = np.uint8(img_msrcp - 1.0)
    return img_msrcp
        
# def wavelet(X):
    # r1 = 2.3
    # r2 = 1.3
    # k = 0.81
    # Kc = 1/6
    # # 一、二级小波分解
    # cA1,(cH1,cV1,cD1) = pywt.dwt2(X,'db4') 
    # cA2,(cH2,cV2,cD2) = pywt.dwt2(cA1,'db4')
    
    # print(type(cA1))
    
    # # LL2 = (r1-r2)(k(x-m)+m)
    # # fun = @(block_struct) (r1-r2)*(k*block_struct.data+(1-k)*mean(mean(block_struct.data)))
    # # cA2 = blockproc(cA2, [2, 2], fun)
    # cA2 = (r1-r2)*(k*cA2+(1-k)*cA2.mean())

    # cA3,(cH3,cV3,cD3) = pywt.dwt2(cA2,'db4')
    
    # print(cA1.shape,cH1.shape,cV1.shape,cD1.shape)
    # print(cA2.shape,cH2.shape,cV2.shape,cD2.shape)
    # print(cA3.shape,cH3.shape,cV3.shape,cD3.shape)

    # sX = X.shape
    # s1 = cA1.shape
    # s2 = cA2.shape
    # # 定义同态滤波器
    # H = np.zeros((3,4)) # H(1,:)一级
    # for j in range(3):
        # for i in range(4):
            # h = i//2
            # v = i%2
            # # fprintf('%d%d\n',h,v)
            # H[j,i] = (r1-r2)/(1+2.415*(((h**2+v**2)**0.5)/(2**j*Kc))**4)
    # # 加权滤波
    # cA3 = cA3*H[2,0]
    # cH3 = cH3*H[2,1]
    # cV3 = cV3*H[2,2]
    # cD3 = cD3*H[2,2]
    # cA2 = cA2*H[1,0]
    # cH2 = cH2*H[1,1]
    # cV2 = cV2*H[1,2]
    # cD2 = cD2*H[1,3]

    # cA1 = cA1*H[0,0]
    # cH1 = cH1*H[0,1]
    # cV1 = cV1*H[0,2]
    # cD1 = cD1*H[0,3]

    # # 逆序重构
    # cA2 = pywt.idwt2((cA3,(cH3,cV3,cD3)),'db4')
    # cA1 = pywt.idwt2((cA2,(cH2,cV2,cD2)),'db4')
    # Xr = pywt.idwt2((cA1,(cH1,cV1,cD1)),'db4')
    
    # return Xr


def test():
    imname = '169620_1.jpg'
    img = cv2.imread(imname, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    ic = IlluminationCorrector(img)
    
    HE_img = ic.HE()
    AHE_img = ic.AHE()
    LT_img = ic.LT()
    GIC_img = ic.GIC()
    LoG_img = LoG(img)
    SSR_img = ic.SSR()
    
    
    img_msrcr = ic.MSRCR()

   
    img_amsrcr = ic.automatedMSRCR()
    img_msrcp = ic.MSRCP()
    
    # HE_img = HE(img)
    # AHE_img = AHE(img)
    # LT_img = LT(img)
    # GIC_img = GIC(img,2)
    # LoG_img = LoG(img)
    # SSR_img = SSR(img)
    
    
    # img_msrcr = MSRCR(
        # img,
        # config['sigma_list'],
        # config['G'],
        # config['b'],
        # config['alpha'],
        # config['beta'],
        # config['low_clip'],
        # config['high_clip']
    # )
   
    # img_amsrcr = automatedMSRCR(
        # img,
        # config['sigma_list']
    # )

    # img_msrcp = MSRCP(
        # img,
        # config['sigma_list'],
        # config['low_clip'],
        # config['high_clip']        
    # ) 
    
    
    plt.figure(figsize=(5, 7),facecolor='gray')
    subplots_adjust(left=0.05,bottom=0.1,top=0.9,right=0.95,hspace=0.05,wspace=0.0)
    
    ax = plt.subplot(3,3,1)
    ax.set_title('ORI',loc='left')
    plt.imshow(img)
    plt.axis('off')
    
    ax = plt.subplot(3,3,2)
    ax.set_title('HE',loc='left')
    plt.imshow(HE_img)
    plt.axis('off')
        
    ax = plt.subplot(3,3,3)
    ax.set_title('AHE',loc='left')
    plt.imshow(AHE_img)
    plt.axis('off')
    
    ax = plt.subplot(3,3,4)
    ax.set_title('GIC',loc='left')
    plt.imshow(GIC_img)
    plt.axis('off')
    
    ax = plt.subplot(3,3,5)
    ax.set_title('LT',loc='left')
    plt.imshow(LT_img)
    plt.axis('off')
    
    ax = plt.subplot(3,3,6)
    ax.set_title('SSR',loc='left')
    plt.imshow(SSR_img)
    plt.axis('off')
    
    ax = plt.subplot(3,3,7)
    ax.set_title('MSRCR',loc='left')
    plt.imshow(img_msrcr)
    plt.axis('off')
    
    ax = plt.subplot(3,3,8)
    ax.set_title('AMSRCR',loc='left')
    plt.imshow(img_amsrcr)
    plt.axis('off')
    
    ax = plt.subplot(3,3,9)
    ax.set_title('MSRCP',loc='left')
    plt.imshow(img_msrcp)
    plt.axis('off')
    
    plt.savefig('contrast.png')
    plt.show()
    
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    
    # cv2.imshow('img', img)
    # cv2.imshow('HE',HE_img)
    # cv2.imshow('AHE',AHE_img)
    # cv2.imshow('log transform', LT_img)
    # cv2.imshow('gamma correction',GIC_img)
    # cv2.imshow('LoG',LoG_img)
    # cv2.imshow('SSR',SSR_img)
    # cv2.imshow('img_msrcr',img_msrcr)
    # cv2.imshow('img_amsrcr',img_amsrcr)
    # cv2.imshow('img_msrcp',img_msrcp)
    # if cv2.waitKey(0) == 27:
        # cv2.destroyAllWindows()


if __name__=='__main__':
    test()