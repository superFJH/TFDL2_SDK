FROM ubuntu:22.04
MAINTAINER Jianhao Feng <feng.jianhao@think-force.com>
LABEL version="3.2.0" description="Thinkforce Service Deploy Docker" by="ThinkForce.Inc"
RUN  sed -i s@/ports.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
# 设置环境变量
ENV LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8

# 安装必要的工具和依赖
ARG TIMEZONE=Asia/Shanghai
RUN ln -fs /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && \
    echo ${TIMEZONE} > /etc/timezone && \
    apt-get update && \
    apt-get install -y \
        locales \
        fonts-wqy-microhei \
        fonts-wqy-zenhei \
        tzdata \
        ssh sshpass vim htop libnuma-dev numactl libx264-dev tzdata libgomp1 gcc g++ cmake python3-pip\
        && rm -rf /var/lib/apt/lists/*

# 配置 Locale 为中文
RUN locale-gen zh_CN.UTF-8 && \
    update-locale LANG=zh_CN.UTF-8

# 设置时区（可选）
#RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
#    dpkg-reconfigure -f noninteractive tzdata


