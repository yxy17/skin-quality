
acne_main.py：痘痘检测主程序，把得到的结果输出成每张图一个txt在bbox_output

eval_tools.py：吧gt_label和自己预测的进行评估

原始标签（gt_label)是（x0,y0,width,height)

acne_main.py输出的标签是（x0,y0,x1,y1)
