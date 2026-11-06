# Windows 环境安装指导（WSL2 + conda）

整个过程大约 30–60 分钟，按顺序做即可，每一步都可以停下来检查。

## 第 1 步：安装 WSL2（Ubuntu）

1. 点开始菜单，搜索 **PowerShell**，右键 → **以管理员身份运行**。
2. 输入：
   ```
   wsl --install -d Ubuntu-22.04
   ```
3. 安装完成后**重启电脑**。重启后会自动弹出 Ubuntu 窗口，按提示设置一个用户名和密码（输入密码时屏幕不显示字符，这是正常的）。

## 第 2 步：（可选但建议）限制 WSL 占用的内存

在 Windows 的文件资源管理器地址栏输入 `%UserProfile%`，新建文本文件 `.wslconfig`，内容（按你的电脑内存调整，下面假设 16 GB）：

```
[wsl2]
memory=10GB
processors=4
```

保存后，在 PowerShell 运行 `wsl --shutdown`，再重新打开 Ubuntu。

## 第 3 步：安装 Miniforge（conda）

在 Ubuntu 窗口里依次输入：

```
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

一路回车、输入 yes。结束后**关闭并重新打开** Ubuntu 窗口。

## 第 4 步：创建计算环境

```
mamba create -n solv -c conda-forge python=3.11 rdkit xtb numpy scipy pandas matplotlib -y
mamba activate solv
pip install pyscf
pip install pyscf-dispersion
```

说明：`pyscf-dispersion` 提供 D3 色散校正，如果它安装失败，可以先跳过，脚本会自动退回到不带色散的版本，并在输出里提醒你。

## 第 5 步：检查安装

```
xtb --version
python -c "import rdkit, pyscf; print(rdkit.__version__, pyscf.__version__)"
```

两条命令都能正常输出版本号，就说明环境好了。如果哪一步报错，把**完整的报错文字**复制给我，我会陪你一起排查。

## 第 6 步：把脚本放进 WSL

Windows 的文件在 Ubuntu 里位于 `/mnt/c/Users/你的用户名/`。建议把脚本复制到 Ubuntu 自己的目录再运行，速度更快：

```
mkdir -p ~/work && cp /mnt/c/Users/你的用户名/Downloads/validate_binding.py ~/work/
cd ~/work
```

## 第 7 步：先做一次很小的试运行

```
mamba activate solv
python validate_binding.py --quick
```

`--quick` 只算 3 个分子，用来确认流程能跑通。把终端的输出和生成的 `results.csv` 发给我就好。
