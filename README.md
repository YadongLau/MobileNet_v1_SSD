# MobileNet_v1_SSD  
声明： 修改后的检测速度并没有多大提高，正在进行后续排查，请谨慎使用。。。。  
  
说明：  
1、backbone使用了mobilenet-v1结构；  
2、ssd的38x38feature map提取自mobilenet网络的第5个dw层；  
3、ssd的19x19feature map提取自mobilenet网络的第13个dw层；  
4、第13层之后的结构没有使用；  
5、第10个dw层未使用；  
6、具体更改见--> MobileNet/mobilenet_v1.py  
  
  
训练部分：  
run one_click.py  
  
如何训练自己的数据集：  
1、本模型直接使用labelme标注好的json文件和对应的bmp结尾的图片，放在同一个文件夹；  
2、然后修改cfg.json中的参数。根据自己的需求进行修改；  
  
预测部分：  
run predict.py  
1、预测指定的路径为上一步生成的文件夹内指向--> train_val_test/test.txt.    
2、还要修改predict下save_path的路径信息，此处作为预测好的照片的存储路径。  
  
Reference  
---------
  
https://github.com/bubbliiiing/ssd-pytorch
 
