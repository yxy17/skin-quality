'''
Evaluation Tools for Pimple, black_head and stain detection.
by Zhong Haoxiang
03/24/2020
'''
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CHANGE YOUR PATH OF RESULT AND GROUND-TRUTH HERE!
task='pimple'
GT_PATH=os.path.join('/data/yd_data/skin-quality/labels',task)
OUR_PATH='TYPE YOUR PATH HERE'
SAVE_PATH='TYPE YOUR PATH HERE'

# IF YOUR RESULT IS ORGANIZED AS [x_leftop,y_lefttop,x_rightbottm,y_rightbottom],set RES_FMT as True
# IF YOUR FORMAT is [x_lefttop,y_lefttop,width,height], set RES_FMT as False
RES_FMT=True



def calculate_IoU(predicted_bound, ground_truth_bound):
    """
    computing the IoU of two boxes.
    Args:
    box: (xmin, ymin, xmax, ymax)
    Return:
    IoU: IoU of box1 and box2.
    """
    p_xmin, p_ymin, p_xmax, p_ymax = predicted_bound
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = ground_truth_bound

    p_area = (p_xmax - p_xmin) * (p_ymax - p_ymin) 
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) 

    # Intersection of two boxes
    xmin = max(p_xmin, gt_xmin)
    ymin = max(p_ymin, gt_ymin)
    xmax = min(p_xmax, gt_xmax)
    ymax = min(p_ymax, gt_ymax)

    w = xmax - xmin
    h = ymax - ymin
    if w <=0 or h <= 0:
        return 0

    area = w * h
    IoU = area / (p_area + gt_area - area)
    return IoU

def singlePictureCalculate(filename,gt_path,pred_path):

    gt_bbox=np.loadtxt(os.path.join(gt_path,filename))
    our_bbox=np.loadtxt(os.path.join(pred_path,filename))
     # If our prediction or groundtruth is none. Save NaN recommended.
     # !!! CAUTION !!!  If np.isnan raises error, CHANGE TO **.size == 1
    if gt_bbox.size==1:
        if our_bbox.size==1:
            return 0,0,0,0
        else:
            our_bbox = our_bbox.reshape(-1, 4)
            return 0,our_bbox.shape[0],1,0

    if our_bbox.size==1:
        gt_bbox = gt_bbox.reshape(-1, 4)
        return gt_bbox.shape[0],0,0,1

    gt_bbox=gt_bbox.reshape(-1,4)
    our_bbox=our_bbox.reshape(-1,4)


    if RES_FMT:
        gt_bbox[:,2:]=gt_bbox[:,2:]+gt_bbox[:,:2]
    else:
        gt_bbox[:, 2:] = gt_bbox[:, 2:] + gt_bbox[:, :2]
        our_bbox[:, 2:] = our_bbox[:, 2:] + our_bbox[:, :2]

    gt_num=gt_bbox.shape[0]
    our_num=our_bbox.shape[0]

    gt_idx_list=[i for i in range(gt_num)]
    our_idx_list=[i for i in range(our_num)]
    success_list=[]

    for gt_idx in gt_idx_list:
        for our_idx in our_idx_list:
            if our_idx in success_list: # if this bbx is already matched, skip
                continue

            iou=calculate_IoU(gt_bbox[gt_idx,:],our_bbox[our_idx,:])
            if iou>0:
                success_list.insert(len(success_list),our_idx)
                break

    success_num=len(success_list)
    false_num=our_num-success_num
    miss_num=gt_num-success_num

    return gt_num,our_num,false_num/our_num,miss_num/gt_num


def allPictureCalculate(gt_path,pred_path,save=True):
    """
        Calculate all pictures' results
        Args:
        gt_path,pred_path,save=True
        Return:
        dataframe of all results, dataframe for positive false, dataframe for negative true
    """
    if os.path.exists(os.path.join(SAVE_PATH,'allEval.csv'))==True:
        print('Results file \'allCalculate.csv\' already exists. Loading...')
        return pd.read_csv(os.path.join(SAVE_PATH,'allEval.csv'),index_col=0),\
               pd.read_csv(os.path.join(SAVE_PATH,'predButNoGT.csv'),index_col=0),\
               pd.read_csv(os.path.join(SAVE_PATH,'GTButNoPred.csv'),index_col=0)

    df=pd.DataFrame()
    df_gt0pred1=pd.DataFrame()
    df_gt1pred0 = pd.DataFrame()
    print('Filename: GT_NUM, OUR_NUM, ERR_RATE, MISS_RATE')

    for filename in os.listdir(gt_path):
        [gt_num,our_num,err_rate,miss_rate]=singlePictureCalculate(filename,gt_path,pred_path)
        print('\r {}: {}, {}, {}, {}'.format(filename,gt_num,our_num,err_rate,miss_rate),end='')
        # valid ones
        df_tmp=pd.DataFrame(data=[[gt_num,err_rate,miss_rate]],index=[filename[:-4]],
                            columns=['GT_NUM','ERR_RATE','MISS_RATE'],dtype=float)
        df=df.append(df_tmp)

        # gt=0, pred!=0
        if gt_num==0 and our_num!=0:
            df_gt0pred1_t=pd.DataFrame(data=[[filename[:-4],our_num]],columns=['FILENAME','OUR_NUM'])
            df_gt0pred1=df_gt0pred1.append(df_gt0pred1_t)

        #gt!=0,pred=0
        if gt_num!=0 and our_num==0:
            df_gt1pred0_t=pd.DataFrame(data=[[filename[:-4],gt_num]],columns=['FILENAME','GT_NUM'])
            df_gt1pred0=df_gt1pred0.append(df_gt1pred0_t)

    if save:
        print('')
        print('Saving results:\n{}\n{}\n{}'.format(os.path.join(SAVE_PATH,'allCalculate.csv'),
                                                   os.path.join(SAVE_PATH,'predButNoGT.csv'),
                                                   os.path.join(SAVE_PATH,'GTButNoPred.csv')))

        df.to_csv(os.path.join(SAVE_PATH,'allEval.csv'))
        df_gt0pred1.to_csv(os.path.join(SAVE_PATH, 'predButNoGT.csv'))
        df_gt1pred0.to_csv(os.path.join(SAVE_PATH, 'GTButNoPred.csv'))

    return df,df_gt0pred1,df_gt1pred0

def evaluate(res_df,df_gt0pred1,df_gt1pred0,show=False,savefig=True):

    df=res_df.drop(index=(res_df.loc[(res_df['GT_NUM']==0)].index))
    df_group=res_df.groupby(by='GT_NUM')
    group_mean=df_group.mean()
    all_mean=res_df.mean()
    all_mean_nozero=df.mean()

    # Draw barplot according to err_rate
    err_order=group_mean.sort_values(by='ERR_RATE',ascending=False).index.to_list()
    fig1=plt.figure()
    ax=sns.barplot(group_mean.index,group_mean['ERR_RATE'],order=err_order)
    ax.set(title='Error rate w.r.t ground truth number')
    if show:
        plt.show()
    if savefig:
        plt.savefig(os.path.join(SAVE_PATH,'ErrRatePlot.png'))
    plt.close(fig1)

    # Draw barplot according to miss_rate
    miss_order=group_mean.sort_values(by='MISS_RATE',ascending=False).index.to_list()
    fig2=plt.figure()
    ax=sns.barplot(group_mean.index,group_mean['MISS_RATE'],order=miss_order)
    ax.set(title='Miss rate w.r.t ground truth number')
    if show:
        plt.show()
    if savefig:
        plt.savefig(os.path.join(SAVE_PATH,'MissRatePlot.png'))
    plt.close(fig2)

    # draw two special situations
    if df_gt0pred1.empty==True:
        print('No zero groundtruth is detected as non-zero!')
    else:
        count_gt0pred1=df_gt0pred1.groupby(by='OUR_NUM').count()
        fig3=plt.figure()
        ax=sns.barplot(count_gt0pred1.index,count_gt0pred1['FILENAME'])
        ax.set(xlabel='Our predicted number',ylabel='Count',title='Count of predicted number while Groudtruth is 0')
        if show:
            plt.show()
        if savefig:
            plt.savefig(os.path.join(SAVE_PATH,'CountOfPredictedNum_NoGT.png'))
        plt.close(fig3)
        count_gt0pred1.rename(columns={'FILENAME': 'COUNT'}, inplace=True)
        count_gt0pred1.sort_values(by='COUNT', ascending=False, inplace=True)
        count_gt0pred1.to_csv(os.path.join(SAVE_PATH, 'CountOfPredictedNum_NoGT.txt'), sep='\t')

    if df_gt1pred0.empty==True:
        print('No non-zero groundtruth is detected as zero!')
    else:
        count_gt1pred0=df_gt1pred0.groupby(by='GT_NUM').count()
        fig4=plt.figure()
        ax=sns.barplot(count_gt1pred0.index,count_gt1pred0['FILENAME'])
        ax.set(xlabel='Groundtruth number',ylabel='Count',title='Count of ground-truth number while prediction is 0')
        if show:
            plt.show()
        if savefig:
            plt.savefig(os.path.join(SAVE_PATH,'CountOfGroundtruthNum_NoPred.png'))
        plt.close(fig4)
        count_gt1pred0.rename(columns={'FILENAME': 'COUNT'}, inplace=True)
        count_gt1pred0.sort_values(by='COUNT', ascending=False, inplace=True)
        count_gt1pred0.to_csv(os.path.join(SAVE_PATH, 'CountOfGroundtruthNum_NoPred.txt'), sep='\t')

    # Save txt file
    all_mean_serie=pd.Series({"ERR_RATE":all_mean['ERR_RATE'],'MISS_RATE':all_mean['MISS_RATE']},name='ALL_MEAN')
    all_mean_nozero_serie=pd.Series({"ERR_RATE":all_mean_nozero['ERR_RATE'],'MISS_RATE':all_mean_nozero['MISS_RATE']},name='ALL_MEAN_NOZERO')
    group_mean=group_mean.append(all_mean_serie)
    group_mean = group_mean.append(all_mean_nozero_serie)
    print('Saving evaluate results:\n{}\n{}\n{}'.format(os.path.join(SAVE_PATH,'EvalResults.txt'),
                                                        os.path.join(SAVE_PATH,'CountOfPredictedNum_NoGT.txt'),
                                                        os.path.join(SAVE_PATH,'CountOfGroundtruthNum_NoPred.txt')))
    group_mean.to_csv(os.path.join(SAVE_PATH,'EvalResults.txt'),sep='\t',float_format='%.3f')



def main():
    df_all,df_gt0pred1,df_gt1pred0=allPictureCalculate(GT_PATH,OUR_PATH)
    evaluate(df_all,df_gt0pred1,df_gt1pred0)


if __name__=='__main__':
    main()
