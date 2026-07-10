# 这是熠知7140 系列芯片的驱动安装目录
## **注意！** 如果是16110平台不需要安装这个目录的驱动，16110平台在出厂时会配置好驱动

## 安装驱动前
'''
请先 apt install gcc g++ cmake build-essential 
'''

## 安装步骤：
1. 解压目录下的压缩包
2. 直接运行里面的insmodTFDriver.sh 就OK了

### 对于视频编解码驱动安装：
直接进入解压后的codec 目录，在里面运行buildTFCoderDriver.sh