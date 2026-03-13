#!/bin/bash
#
# Nougat Data Engineering — 一键环境配置脚本
# 用法: bash setup_env.sh [环境名称，默认 nougat]
#
set -e

ENV_NAME="${1:-nougat}"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_RED='\033[0;31m'
C_RESET='\033[0m'

ok()   { echo -e "${C_GREEN}✓ $1${C_RESET}"; }
warn() { echo -e "${C_YELLOW}⚠ $1${C_RESET}"; }
fail() { echo -e "${C_RED}✗ $1${C_RESET}"; }

echo "========================================"
echo " Nougat Data Engineering — 环境配置"
echo " 目标 conda 环境: $ENV_NAME"
echo "========================================"

# -------- [1/7] conda --------
echo ""
echo "[1/7] 检查 conda..."
if ! command -v conda &> /dev/null; then
    fail "conda 未找到。请先安装 Miniconda/Anaconda。"
    exit 1
fi
ok "conda $(conda --version 2>&1 | awk '{print $2}')"

# -------- [2/7] 创建环境 --------
echo ""
echo "[2/7] 创建/激活 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)..."
if conda env list 2>/dev/null | grep -qw "$ENV_NAME"; then
    echo "    环境 '$ENV_NAME' 已存在，跳过创建。"
else
    conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
ok "环境已激活: $(python --version)"

# -------- [3/7] PyTorch --------
echo ""
echo "[3/7] 安装 PyTorch..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
    ok "PyTorch $TORCH_VER 已安装 (CUDA=$CUDA_OK)"
else
    if command -v nvidia-smi &> /dev/null; then
        echo "    检测到 NVIDIA GPU，安装 CUDA 版 PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "    未检测到 GPU，安装 CPU 版 PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    ok "PyTorch 安装完成"
fi

# -------- [4/7] nougat + dataset 依赖 --------
echo ""
echo "[4/7] 安装 nougat 及数据集依赖..."
pip install -e "${SCRIPT_DIR}[dataset]"
pip install jieba
ok "nougat + dataset 依赖已安装"

# -------- [5/7] LaTeXML --------
echo ""
echo "[5/7] 检查 LaTeXML..."
if command -v latexml &> /dev/null; then
    ok "latexml 已安装"
else
    warn "latexml 未找到。预处理流水线 (preprocess_pipeline) 需要它。"
    echo "    安装方法:"
    echo "      Ubuntu/Debian: sudo apt-get install latexml"
    echo "      macOS:         brew install latexml"
    echo "      CPAN:          cpanm LaTeXML"
    echo "    文档: https://math.nist.gov/~BMiller/LaTeXML/"
fi

# -------- [6/7] pdffigures2 / Java --------
echo ""
echo "[6/7] 检查 pdffigures2..."
if [ -n "$PDFFIGURES_PATH" ] && [ -f "$PDFFIGURES_PATH" ]; then
    ok "pdffigures2 JAR: $PDFFIGURES_PATH"
else
    if command -v java &> /dev/null; then
        warn "PDFFIGURES_PATH 未设置或 JAR 不存在。"
        echo "    Java 已安装: $(java -version 2>&1 | head -1)"
    else
        warn "Java 未安装。pdffigures2 需要 Java 运行时。"
    fi
    echo "    步骤:"
    echo "      1. git clone https://github.com/allenai/pdffigures2 && cd pdffigures2"
    echo "      2. sbt assembly  (需安装 sbt)"
    echo "      3. export PDFFIGURES_PATH=/path/to/pdffigures2.jar"
fi

# -------- [7/7] 汇总 --------
echo ""
echo "[7/7] 环境总览"
echo "========================================"
echo "  Conda 环境:  $ENV_NAME"
echo "  Python:      $(python --version 2>&1)"
python -c "import torch; print(f'  PyTorch:     {torch.__version__}  (CUDA={torch.cuda.is_available()})')" 2>/dev/null || echo "  PyTorch:     未安装"
python -c "import nougat; print('  Nougat:      已安装')" 2>/dev/null || echo "  Nougat:      未安装"
command -v latexml &>/dev/null && echo "  LaTeXML:     已安装" || echo "  LaTeXML:     未安装 (预处理需要)"
[ -n "$PDFFIGURES_PATH" ] && [ -f "$PDFFIGURES_PATH" ] && echo "  pdffigures2: $PDFFIGURES_PATH" || echo "  pdffigures2: 未配置"
echo ""
echo "  激活环境: conda activate $ENV_NAME"
echo "========================================"
