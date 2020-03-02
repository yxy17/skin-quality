#coding:utf-8
import os
import pandas as pd

input_file = 'label/skin_type.txt'
# file = open(input_file,'r')
# data = pd.read_csv(input_file)
# print(data.info())


#skin_color = ['透白','白皙','自然','小麦','暗沉','黝黑']
skin_color = ['油性','干性','中性','混合性']
#corr_label = [0,1,2,3,4,5]


my_dict = {}
my_dict['path']=[]
my_dict['label'] = []

print("my_dict",my_dict)
k = 0
with open(input_file,encoding='UTF-8') as f:
    for i, line in enumerate(f.readlines()):
        k = k+1
        #if k >5:
         #   break

        z = line.split("\t")
        image_name = '/data/xujia/bounded_skin_data/imgs/'+z[0]+'.jpg'
        #image_name = '/data/yd_data/skin-quality/imgs/' + z[0] + '.jpg'
        #image_name = z[0] + '.jpg'
        zw_type = z[1].strip()
        label = skin_color.index(zw_type)
        print(image_name,zw_type,label)

        my_dict['path'].append(image_name)
        my_dict['label'].append(label)
        # i 是行数
        # line 是每行的字符串，可以使用 line.strip().split(",") 按分隔符分割字符串
    f.close()

df = pd.DataFrame(my_dict)

df.to_csv("label/ori_skin_type_label.txt",index=False)