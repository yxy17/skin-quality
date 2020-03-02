#coding=utf-8
#!python3
# parse the json file

import os
import json
import jsonlines
import requests
ROOT = '/data/libo/ydskin/'     # json文件的目录
FILEPATH = ROOT + 'face_data_20200218.json'

PICPATH = ROOT + 'imgs'         # 下载的图像的目录
LABELPATH = ROOT+'labels'       # 生成的label的目录

import logging
logging.basicConfig(filename = 'mylog.txt',level=logging.DEBUG, format=' %(asctime)s - %(levelname)s -   %(message)s') 
logging.disable(logging.DEBUG) 

# 通过记录中的url字段下载图像
def downloadPicByUrl(url,path):

    r = requests.request('get',url) # 
    #print(r.status_code)
    with open(path,'wb') as f: 
        f.write(r.content)
    f.close()

#  json文件预处理
def presub(filepath):
    f = open(filepath, 'r', encoding='utf-8')
    lines = f.read()
    f.close()
    
    print(lines[-10:])
    
    lines_json = lines.replace('jpg\"}{\"','jpg\"}\n{\"')   # 将原先无逗号分割的单行json数据替换为每行包含一条数据的新json文件方便处理
    
    newpath = filepath[:-5]+'_new'+filepath[-5:]            #  新json文件重命名
    logging.debug('filepath: %s'%newpath)
    
    n = open(newpath, 'w', encoding='utf-8')
    n.write(lines_json)
    n.close()
    return newpath                                          #   返回新json文件的目录
 
##  处理stain,black head和pimple
def stainEtc(report_id,value,rectangle,TXT): 
    TXT.write('%d\t%d\n'%(report_id,value))                 #   report_id为图像id,value为色斑（黑头、痘痘）个数
    for i in range(value):                                  #   写入每个色斑（黑头、痘痘）的bounding box信息
        # logging.info('width\t%s\n'%type(rectangle[i]['width']))
        # logging.info('top\t%s\n'%type(rectangle[i]['top']))
        # logging.info('height\t%s\n'%type(rectangle[i]['height']))
        # logging.info('left\t%s\n'%type(rectangle[i]['left']))
        TXT.write('%f\t\t%f\t\t%f\t\t%f\n'%(rectangle[i]['width'],rectangle[i]['top'],rectangle[i]['height'],rectangle[i]['left']))
    TXT.write('\n')

##  处理skin color和skin type
def skinColorType(report_id,skin_value,TXT):
    TXT.write('%d\t%s\n'%(report_id,skin_value))

##  处理skin age
def skinAge(report_id,skin_age,TXT):
    TXT.write('%d\t%d\n'%(report_id,skin_age))

##  处理rose_acne
def roseAcne(report_id,rose_acne,TXT):
    ret = [0]*5
    for i in range(len(rose_acne)):
        key = rose_acne[i]['type']
        val = rose_acne[i]['value']
        ret[key] = val
    #print(ret)
    TXT.write('%d\t'%report_id)
    for i in range(5):
        TXT.write('%d '%ret[i])
    TXT.write('\n')
    
##  处理coarse pore
def coarsePore(report_id,coarse_pore,TXT):
    ret = [0]*4
    for i in range(len(coarse_pore)):
        key = coarse_pore[i]['type']
        val = coarse_pore[i]['value']
        ret[key] = val
    #print(ret)
    TXT.write('%d\t'%report_id)
    for i in range(4):
        TXT.write('%d '%ret[i])
    TXT.write('\n')

##  处理wrinkle
def wrinkle(report_id,wrinkle_detail,TXT):
    ret_val = [0]*5
    ret = [[0,0],[0,0],[0,0],[0,0]]
    for item in wrinkle_detail:
        
        if(item['type']==0):
            ret_val[0] = item['value']
        else:
            ret_val[item['type']] = item['value']
            for t in item['detail']:
                ret[item['type']-1][t['detail_type']] = t['value']

    
    TXT.write('%d\n'%report_id)
    for i in ret_val:
        TXT.write('%d '%i)
    TXT.write('\n')
    for i in range(4):
        for j in ret[i]:
            TXT.write('%d '%j)
        TXT.write('\n')
    TXT.write('\n')

##  处理black eye
def blackEye(report_id,black_eye_detail,TXT):
    ret = [0,0]
    ret_val = [[0,0],[0,0]]
    for item in black_eye_detail:

        ret[item['type']] = item['value']
        for t in item['detail']:
            # print(t)
            # print(type(t))
            ret_val[item['type']][0] = t['detail_type']
            ret_val[item['type']][1] = t['level']

    
    TXT.write('%d\n'%report_id)
    for i in ret:
        TXT.write('%d '%i)
    TXT.write('\n')
    for i in range(2):
        for j in ret_val[i]:
            TXT.write('%d '%j)
        TXT.write('\n')
    TXT.write('\n')

def processjson(filepath):
    with open(filepath, "r+", encoding="utf8") as f:
        line = 0
        
        ## 创建10个txt文件用于存储labels
        stainTXT = open(os.path.join(LABELPATH,'stain.txt'),'w',encoding="utf8")
        blackHeadTXT = open(os.path.join(LABELPATH,'black_head.txt'),'w',encoding="utf8")
        pimpleTXT = open(os.path.join(LABELPATH,'pimple.txt'),'w',encoding="utf8")
        skinTypeTXT = open(os.path.join(LABELPATH,'skin_type.txt'),'w',encoding="utf8")
        skinColorTXT = open(os.path.join(LABELPATH,'skin_color.txt'),'w',encoding="utf8")
        skinAgeTXT = open(os.path.join(LABELPATH,'skin_age.txt'),'w',encoding="utf8")
        roseAcneTXT = open(os.path.join(LABELPATH,'rose_acne.txt'),'w',encoding="utf8")
        coarsePoreTXT = open(os.path.join(LABELPATH,'coarse_pore.txt'),'w',encoding="utf8")
        wrinkleTXT = open(os.path.join(LABELPATH,'wrinkle.txt'),'w',encoding="utf8")
        blackEyeTXT = open(os.path.join(LABELPATH,'black_eye.txt'),'w',encoding="utf8")
            
        for item in jsonlines.Reader(f):    #   逐行读取处理后的json文件，每一行包含了一张图像的所有标签信息
            
            ## test
            # print(item['report_id'])
            # continue
        
            ## fixed
            photo_url = item['photo_url']   #   获取图像的url用于下载，
            report_id = item['report_id']   #   获取图像的report_id,为int类型
            
            ## 
            ## log
            line = line + 1
            logging.info('%d\t%d'%(line,report_id))
            
            print('%d\t%d'%(line,report_id))    #   输入处理的图像的序号的reort_id

            ## 读取10个子任务的具体标签信息
            stain_detail = item['stain_detail']             
            black_head_detail = item['black_head_detail']
            pimple_detail = item['pimple_detail']

            skin_type = item['skin_type']
            skin_color = item['skin_color']
            skin_age = item['skin_age']

            coarse_pore_detail = item['coarse_pore_detail']
            rose_acne_detail = item['rose_acne_detail']

            wrinkle_detail = item['wrinkle_detail']
            black_eye_detail = item['black_eye_detail']
            
            
            
            ##  下载图像，后续只用于生成labelse时可注释掉，无需重复下载
            picpath = os.path.join(PICPATH,'%d.jpg'%report_id)
            logging.debug('photo_url: %s'%photo_url)
            logging.debug('report_id: %d'%report_id)
            logging.debug('%s'%picpath)
            downloadPicByUrl(photo_url,picpath)  # download pictures named by report_id
            
            ##  stain
            stain_detail = json.loads(stain_detail)     ##  stain_detail原本为str格式，转化为字典
            if(not stain_detail):                       ##  若为空字典，则色斑个数为0，bounding box为空列表
                value = 0
                rectangle = []
            else:
                rectangle = stain_detail['rectangle']   ## type list 
                value = stain_detail['value']           ## type int
            stainEtc(report_id,value,rectangle,stainTXT)
            
            ##  black_head  与stain一致
            black_head_detail = json.loads(black_head_detail)
            if(not black_head_detail):
                value = 0
                rectangle = []
            else:
                rectangle = black_head_detail['rectangle']   ## type list 
                value = black_head_detail['value']           ## type int
            stainEtc(report_id,value,rectangle,blackHeadTXT)
            
            ## pimple       与stain一致
            pimple_detail = json.loads(pimple_detail)
            
            if(not pimple_detail):
                value = 0
                rectangle = []
            else:
                rectangle = pimple_detail['rectangle']   ## type list 
                value = pimple_detail['value']           ## type int
            stainEtc(report_id,value,rectangle,pimpleTXT)
            
            logging.info('stain black head and pimple processed')
            
            ## skin_type
            skinColorType(report_id,skin_type,skinTypeTXT)
            ## skin color
            skinColorType(report_id,skin_color,skinColorTXT)
            ## skin_age
            skinAge(report_id,skin_age,skinAgeTXT)
            
            logging.info('skin type color and age processed')
            
            ## rose_acne
            rose_acne_detail = json.loads(rose_acne_detail)
            roseAcne(report_id,rose_acne_detail,roseAcneTXT)
            
            ## coarse_pore
            coarse_pore_detail = json.loads(coarse_pore_detail)
            coarsePore(report_id,coarse_pore_detail,coarsePoreTXT)
            
            logging.info('coarse pore and rose acne processed')
                
            ## wrinkle
            wrinkle_detail = json.loads(wrinkle_detail)
            wrinkle(report_id,wrinkle_detail,wrinkleTXT)
            
            ## black_eye

            # print(black_eye_detail)
            black_eye_detail = json.loads(black_eye_detail)
            blackEye(report_id,black_eye_detail,blackEyeTXT)
            
            logging.info('wrinkle and black eye processed')
         
        stainTXT.close()
        blackHeadTXT.close()
        pimpleTXT.close()
        skinTypeTXT.close()
        skinColorTXT.close()
        skinAgeTXT.close()
        roseAcneTXT.close()
        coarsePoreTXT.close()
        wrinkleTXT.close()
        blackEyeTXT.close()        


if __name__ == '__main__':
    NEWPATH = presub(FILEPATH)
    processjson(NEWPATH)
    
    
    
