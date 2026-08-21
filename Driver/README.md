# 这是熠知7140 系列芯片的驱动安装目录
## **注意！** 如果是16110平台不需要安装这个目录的驱动，16110平台在出厂时会配置好驱动

## 安装驱动前
'''
请先 apt install gcc g++ cmake build-essential 
'''
# 1.老版本驱动，只支持4G reserved DDR 空间作为NPU的使用空间不可以扩展

## 安装步骤：
1. 解压olddriver目录下的压缩包
2. 直接运行里面的insmodTFDriver.sh 就OK了

### 对于视频编解码驱动安装：
直接进入解压后的codec 目录，在里面运行buildTFCoderDriver.sh

# 2，新版本驱动，支持使用tf_hugepage_register 工具申请Hugepage ddr 动态扩展NPU的内存空间，从而塞进去更多权重，如果是视频编解码驱动还是使用olddriver里面的codec

## 安装步骤：
直接运行Driver下的insmodTFDriver.sh 就OK了
