'''
landmarks detection and save for yd_skin-quality
copyright: Haoxiang Zhong
time: 03/20/2020
'''

import numpy as np
import face_alignment,os
from skimage import io,transform

ROOT='/data/yd_data/skin-quality/bounded_skin_data/imgs'
SAVE_DIR='/data/yd_data/skin-quality/landmarks/txt'
ROTATE_LIST='/data/yd_data/skin-quality/landmarks/rotate_list.txt'
TOOSMALL_LIST='/data/yd_data/skin-quality/landmarks/toosmall_list.txt'

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

for filename in os.listdir(ROOT):
    save_file=filename[:-4]+'.txt'
    if os.path.exists(os.path.join(SAVE_DIR,save_file)):
        print('\r Already processed, skipping... {} '.format(filename),end='')
        continue
    
    # print current img info
    print('\r Current img: {} '.format(filename),end='')
    
    im=io.imread(os.path.join(ROOT,filename))
    resized=False
    scale=1.0

    # Process imgs if it's too huge or rotated
    [im_height, im_width, im_channel] = im.shape
    if (im_height < im_width): # rotate the picture if it is horizontally placed
        im = transform.rotate(im, 90, resize=True)
        # record rotated img list
        with open(ROTATE_LIST,'a+') as f:
            f.write(filename+'\n')
        
    # too small
    if im_height <= 80:
        with open(TOOSMALL_LIST,'a+') as f:
            f.write(filename+'\n')
        with open(os.path.join(SAVE_DIR,save_file),'w') as f:
            f.write('NaN')
        continue
    
    if (im_height >= 800):
        scale = 800 / im_height
        im = transform.resize(im, (int(im_height * scale), int(im_width * scale), im_channel))
        resized=True

    # get_landmark
    preds=fa.get_landmarks(im)

    #save landmark
    if preds==None:
        with open(os.path.join(SAVE_DIR,save_file),'w') as f:
            f.write('NaN')
    else:
        preds=preds[0]
        if resized==True:
            preds=np.fix(preds/scale)
        np.savetxt(os.path.join(SAVE_DIR,save_file),preds,fmt='%f')



