## git 仓库使用

### git clone

1. 进入 Linux 某文件夹并克隆仓库

```bash
git clone https://github.com/ssh-hyx-ysb/mathematical-modeling.git
```

2. 在 VS code 下打开 WSL 链接

3. 点击`文件` -> `打开文件夹..` -> 选择并打开仓库文件夹

### git commit

- 提交之前需要配置用户名和提交邮箱

```bash
git config --global user.name [GitHub 用户名]
git config --global user.email [GitHub 提交邮箱]
```

- VS code 提交方式

当有文件产生修改（创建、修改、删除、忽略等）时，看到 VS code 左侧按钮（源代码管理按钮）有提示，点进去，输入消息（一定要输入！！！），然后点提交，然后点同步即可

- 命令行提交方式

```bash
git add [filename]  # 暂存文件更改
git commit -m "提交消息" # 提交暂存文件，提交消息为"提交消息"
git push # 推送到远程仓库
```

### git pull

- VS code 拉取方式

在 VS code 源代码管理栏，点击同步（远程仓库有修改的情况下）即可拉取远程仓库的更改，或点击与`更改`在同一行的三个点按钮，点`拉取`手动同步远程仓库

- 命令行拉取方式

```bash
git pull    # 拉取远程仓库的修改
```

## VS code 使用

### 插件推荐

```plain
Error Lens          # 更好的报错显示
vsode-icons         # 更好的图标显示
Lingma - Alibaba    # 通义灵码，AI辅助编程工具
Typora              # Markdown 文件编辑支持
LaTeX Workshop      # LaTeX VS code 支持
Python              # Python VS code 支持
Pylance             # Python VS code 支持
Python Debugger     # Python VS code 支持
Python Environments # Python VS code 支持
Black Formatter     # Python 文件格式化插件
Jupyter Slide Show  # Jupyter VS code 支持
Jupyter PowerToys   # Jupyter VS code 支持
Jupyter Cell Tags   # Jupyter VS code 支持
Jupyter Keymap      # Jupyter VS code 支持
Jupyter Hub         # Jupyter VS code 支持
```

## Latex 模板使用

### Latex 下载

```bash
sudo apt install texlive-full
```

### 基础语法（基于仓库内模板）

- 字体

```latex
\textbf{粗体文本}
\textit{斜体文本}
\underline{带下划线的文本}
```

- 摘要和关键词

```latex
\begin{abstract}                    % 摘要起始

    \textbf{针对问题一，}           % 问题一

    \textbf{针对问题二，}           % 问题二

    \textbf{针对问题三，}           % 问题三

    \textbf{针对问题四，}           % 问题四

    \keywords{word1 \quad word2}    % 关键词
\end{abstract}                      % 摘要结束
```

- 章节标题

```latex
\section{章节名}
```

- 小节标题

```latex
\subsection{小节名}
```

- 小节内小节标题

```latex
\subsubsection{小节内小节名}
```

- 公式

```latex
XXX$行内公式$XXX

\begin{equation}
段公式
\end{equation}
```

- 列表

```latex
\begin{itemize}
    \item 第一项
    \item 第二项
\end{itemize}

\begin{enumerate}
    \item 第一步
    \item 第二步
\end{enumerate}
```

- 图

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{图片路径} % 以70%文件宽引用并显示位于`图片路径`下的图片
    \caption{图片标题}  % 显示在图片下的标题
    \label{图片标签}    % 引用时显示此图片的序号
\end{figure}
```

- 表

```latex
\begin{table}[H]
    \centering
    \caption{表标题}
    \label{表标签}
    \begin{tabular}{表型}
    \toprule    % 上表线
    表头
    \midrule    % 表头表身分割线
    内容
    \bottomrule % 下表线
    \end{tabular}
    \begin{tablenotes}[para,small]
        % 表注
    \end{tablenotes}
\end{table}
```

- 引用

```latex
\ref{xx}    % 引用label为`xx`的图或表
\cite{xx}   % 引用文献关键字为`xx`的文献
```

## uv 使用

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
sudo mv CrashCore /usr/bin/clash
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

```conf
[Unit]
Description=Clash

[Service]
Restart=no
ExecStart=-/usr/bin/clash

[Install]
WantedBy=multi-user.target
```

6. 移动文件到 systemd 文件夹

```bash
sudo mv clash.service /etc/systemd/system/
```

7. 刷新 systemd 缓存

```bash
sudo systemctl daemon-reload
```

8. 创建 config 文件夹

```bash
sudo mkdir -p /.config/clash
```

9. 将刚刚下载的 config.yaml 移动到这个文件夹（假设 config.yaml 在当前文件夹下）

```bash
sudo mv config.yaml /.config/clash
```

10. 启动 clash 服务

```bash
sudo systemctl satrt clash      # 第一次运行下载必须的文件
sudo systemctl restart clash    # 第二次运行正常启动
sudo systemctl enable clash     # 添加clash开机自启动
```

11. 添加代理

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
