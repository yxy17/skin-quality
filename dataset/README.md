# parseJsonFile.py
  
**需要安装jsonlines模块**  
`conda install jsonlines`    
  
以下为此py文件中各函数的说明，参考代码中注释，以及脸部检测的word文档

## 1.label输出函数  
   
  **处理每张图像的标签用到的函数如下，在以下函数内部可修改输出label的格式**  
  
  ``` 
  stainEtc(report_id,value,rectangle,TXT)  
  skinColorType(report_id,skin_value,TXT)  
  skinAge(report_id,skin_age,TXT)  
  roseAcne(report_id,rose_acne,TXT)  
  coarsePore(report_id,coarse_pore,TXT)  
  wrinkle(report_id,wrinkle_detail,TXT)  
  blackEye(report_id,black_eye_detail,TXT)  
  ```

  
## 2.处理流程函数
```
processjson(filepath)  
##逐行读取json文件，每个循环处理一张图片，包括下载，生成每个子任务的label等
```

## 3.json预处理及图像下载函数  

**json预处理是把原本一行包含了所有数据的文件，改为了每行包含一张图像的所有信息的格式，生成了新的json文件**  
  
**图像下载函数在仅仅生成标签时无需使用**
```  
downloadPicByUrl(url,path)  ##用于下载图像，仅生成label时，可在processjson函数中将其注释
presub(filepath)            ##用于对原始json文件预处理，可直接使用提供的新json文件，  
                            ##将主函数中的presub注释，并修改FILEPATH为新的json文件的目录
```
## 4.Usage  
在py文件内部作所需的修改后直接运行  
`python parseJsonFile.py`

