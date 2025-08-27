## git 仓库使用

1. 进入 Linux 某文件夹并克隆仓库

```bash
git clone https://github.com/ssh-hyx-ysb/mathematical-modeling.git
```

2. 在 VS code 下打开 WSL 链接

3. 点击`文件` -> `打开文件夹..` -> 选择并打开仓库文件夹

## Latex 模板使用

### Latex 下载

```bash
sudo apt install texlive-full
```

## UV 使用

### 下载安装

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # 下载uv，需配置代理
source $HOME/.local/bin/env                         # 配置环境变量
```

### 环境初始化

```bash
uv init                 # 初始化当前文件夹的uv环境
uv init [PROJECT_NAME]  # 在当前文件夹下初始化一个名为[PROJECT_NAME]的uv环境
```

### 环境加载

在一个有`uv.lock`和`pyproject.toml`文件的文件夹下，运行这个命令可以快速搭建起可用的 Python 环境

```bash
uv sync
```

### 添加 pip 包

```bash
uv pip install PACKAGE_NAME
```

或

```bash
uv add PACKAGE_NAME
```

### 运行.py 文件

```bash
uv run xxx.py
```

## Linux Clash 代理使用

1. 访问下面的代理网站，注册账号，购买套餐,在`下载和教程`->`Linux`->`2.`下面找到 config.yaml 下载地址，直接复制并下载，然后重命名此文件为 config.yaml 并拖入 Linux 文件夹

```url
https://ikuuu.de
```

2. 访问 Linux Clash 启动器下载链接下载 Clash Linux 启动器

```url
 https://github.com/DustinWin/proxy-tools/releases/download/Clash-Premium/clashpremium-release-linux-amd64.tar.gz
```

3. 解压并放入 Linux 文件夹，并在此文件夹下赋予可执行权限

```
tar -xzvf clashpremium-release-linux-amd64.tar.gz
chmod +x CrashCore
mv CrashCore /usr/bin/clash
```

4. 创建 clash.service 文件

```bash
touch clash.service
```

5. 写入内容

使用`nano`文件编辑器打开刚刚创建的文件

```bash
nano clash.service
```

写入以下内容

```nano
[Unit]
Description=Clash

[Service]
Restart=no
ExecStart=-/usr/bin/clash

[Install]
WantedBy=multi-user.target
```

6. 刷新 systemctl 缓存

```bash
sudo systemctl daemon-reload
```

7. 创建 config 文件夹

```bash
sudo mkdir -p /.config/clash
```

7. 将刚刚下载的 config.yaml 移动到这个文件夹（假设 config.yaml 在当前文件夹下）

```bash
sudo mv config.yaml /.config/clash
```

8. 启动 clash 服务

```bash
sudo systemctl satrt clash      # 第一次运行下载必须的文件
sudo systemctl restart clash    # 第二次运行正常启动
sudo systemctl enable clash     # 添加clash开机自启动
```

9. 添加代理

- 打开主文件夹下的`.bashrc`文件

```bash
nano ~/.bashrc
```

- 在最后添加以下内容

```sh
export http_proxy="http://127.0.0.1:7890"
export Http_Proxy="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export Https_Proxy="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
```

- 刷新缓存

```bash
source ~/.bashrc
```
