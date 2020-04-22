# skin-quality

There are 10 sub-tasks in skin quality.

#classification
  (1)skin age : [1-10,11-20, 21-30, 31-40, 41-50, 51-60, 60+] (7 classes)
  (2)skin color : [透白，白皙，自然，小麦，暗沉，黝黑] （7 classes）
  (3)skin quality: [油性，干性，中性，混合]
  
#little object detections:
  (4)痘痘：[0，1-10,11-20, 21-30, 31-40, 41-50, 51-60, 60+] (7 classes)
  (5)黑头：[0，1-10,11-20, 21-30, 31-40, 41-50, 51-60, 60+] (7 classes)
  (6)色斑：[0，1-10,11-20, 21-30, 31-40, 41-50, 51-60, 60+] (7 classes)

#multi-attribute binary classification
  (7)dark circles:
                色素型 血管型 阴影型 
    left eye     0/1   0/1   0/1
    right eye    0/1   0/1   0/1
  
  (8)Wrinkle：
           鱼尾纹  眼纹  法令纹  抬头纹
    left    0/1   0/1   0/1   (0/
    righ    0/1   0/1   0/1   ( 1 )
    
  (9)Cross pore(毛孔粗大)：
    left face  |   right face  |  brow(眉间) |  forehead（前额）
        0/1              0/1         0/1           0/1
   
  (10)Rosacea(玫瑰痤疮)：
    left face  |   right face  |  Nasal region(鼻周) |  forehead（前额）|  chin（下巴）
        0/1              0/1             0/1                0/1            0/1
 the 
    
