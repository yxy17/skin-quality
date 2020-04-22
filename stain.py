#!python3
#coding=utf-8
import numpy as np
import pywt
import cv2
import os   
from illumination_correction import *
from pylab import *

class Stain:
    def __init__(self,inputDir = './input',outputDir = './output'):
        self.inputDir = inputDir
        self.outputDir = outputDir
    
    # OTSU肤色检测算法
    def skinMask(self,img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)    
        (y, cr, cb) = cv2.split(ycrcb)
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        dst = cv2.bitwise_and(img, img, mask=skin)    
        return skin,dst
    # 计算肤色部分均值
    def skinMean(self,face_array,mask):
        hR,hG,hB = face_array.sum((0,1))  ## all face
        m = mask.sum((0,1))/255 # number of face pixels
        hR, hG, hB = hR/m, hG/m, hB/m
        hR, hG, hB = round(hR,3), round(hG,3), round(hB,3)
        return hR,hG,hB

    def threshBinary(self,X,min_val,max_val):
        mask = (X>min_val)*(X<max_val)
        return mask

    def canny(self,img):
        img_edges=cv2.Canny(img,80,80)
        return img_edges

    def processSingleImage(self,img):
        # 调用光照矫正算法
        #ic = IlluminationCorrector(img)
        #img = ic.HE()

        (r,g,b) = cv2.split(img)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        skin,_ = self.skinMask(img)
        y_mean,cr_mean,cb_mean = self.skinMean(ycrcb,skin)
        (Y, cr, cb) = cv2.split(ycrcb)

        #mask1 = self.threshBinary(cr,cr_mean-20,cr_mean+20)  # cr的取值
        #mask2 = self.threshBinary(cb,cb_mean-14,cb_mean+14)  # cb的取值
        mask1 = self.threshBinary(cr,133,173)
        mask2 = self.threshBinary(cb,77,127)

        # 边缘检测
        img_edges = self.canny(img)
        # 膨胀对边缘进行膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        dilated = cv2.dilate(img_edges,kernel)
        
        ret,thresh = cv2.threshold(dilated,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        mask4 = np.zeros(img.shape,np.uint8)
        for h,cnt in enumerate(contours):
            cv2.drawContours(mask4,[cnt],0,(255,255,255),-1)
        
        mask4=cv2.cvtColor(mask4,cv2.COLOR_BGR2GRAY)

        mask3 = (Y<(y_mean-50))*(b<100)*(g<150)*(r<150)   # Y和rgb的取值的mask
        mask4 = (mask4>0)                                 # 边缘检测的mask
        
        mask = mask1*mask2*mask3*mask4*255;
        mask1 = np.uint8(mask1*255)
        mask2 = np.uint8(mask2*255)
        mask3 = np.uint8(mask3*255)
        mask4 = np.uint8(mask4*255)
        
        
        mask = np.uint8(mask)
        contours,hier = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        ret = img.copy()
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            #print(x,y,w,h)
            cv2.rectangle(ret,(x,y),(x+w,y+h),(0,0,255),2)
        
        mask = np.uint8(255-mask)
        return mask,ret,contours
    
    def processImages(self):
        ImageList = os.listdir(self.inputDir)
        if not os.path.exists(self.outputDir):
            os.mkdir(self.outputDir)
        
        for imgName in ImageList:
            imgPath = os.path.join(self.inputDir,imgName)
            img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            mask,ret,contours = self.processSingleImage(img)
            
            outputPath = os.path.join(self.outputDir,imgName.split('.')[0]+'.txt')
            outputFile = open(outputPath,'w')

            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                outputFile.write('%d\t%d\t%d\t%d\t\n'%(x,y,w,h))
            outputFile.close()
            
def main():
    stain = Stain()
    stain.processImages()
    
if __name__ == '__main__':
    main()