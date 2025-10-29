# Kronos 项目操作手册

## 目录
1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [模型预测使用](#模型预测使用)
4. [模型微调](#模型微调)
5. [回测评估](#回测评估)
6. [常见问题解决](#常见问题解决)

## 环境准备

### 系统要求
- **操作系统**：Windows 10/11, Linux, macOS
- **Python版本**：Python 3.10+
- **GPU**：推荐NVIDIA GPU（支持CUDA）
- **内存**：建议16GB+

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
```

#### 2. 创建虚拟环境
```bash
# 使用conda
conda create -n kronos python=3.10
conda activate kronos

# 或使用venv
python -m venv kronos_env
# Windows
kronos_env\Scripts\activate
# Linux/macOS
source kronos_env/bin/activate
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 快速开始

### 基础预测示例

#### 运行示例脚本
```bash
cd examples
python prediction_example.py
```

#### 手动执行预测
```python
import pandas as pd
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# 1. 加载模型和分词器
from pathlib import Path
tokenizer = KronosTokenizer.from_pretrained(Path(r"E:\code\Kronos\Kronos-Tokenizer-base"))
model = Kronos.from_pretrained(Path(r"E:\code\Kronos\Kronos-small"))

# 2. 创建预测器
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. 准备数据
df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 4. 设置参数
lookback = 400      # 回看历史数据长度
pred_len = 120      # 预测未来数据长度

# 5. 准备输入
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 6. 执行预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # 温度参数
    top_p=0.9,      # 核采样概率
    sample_count=1  # 采样次数
)

print("预测结果前5行：")
print(pred_df.head())
```

## 模型预测使用

### 预测器参数详解

#### KronosPredictor 初始化参数
```python
predictor = KronosPredictor(
    model,           # Kronos模型实例
    tokenizer,       # KronosTokenizer实例
    device="cuda:0", # 设备选择
    max_context=512  # 最大上下文长度
)
```

#### predict 方法参数
```python
pred_df = predictor.predict(
    df=x_df,                    # 历史数据DataFrame
    x_timestamp=x_timestamp,    # 历史时间戳
    y_timestamp=y_timestamp,    # 预测时间戳
    pred_len=pred_len,         # 预测长度
    T=1.0,                     # 温度参数（控制随机性）
    top_p=0.9,                 # 核采样概率
    sample_count=1,            # 采样次数
    verbose=True               # 是否显示进度
)
```

### 数据格式要求

#### 输入数据格式
```python
# 必需列
required_columns = ['open', 'high', 'low', 'close']

# 可选列
optional_columns = ['volume', 'amount']

# 数据示例
df = pd.DataFrame({
    'open': [100.0, 101.0, 102.0],
    'high': [101.5, 102.5, 103.0],
    'low': [99.5, 100.5, 101.5],
    'close': [101.0, 102.0, 102.5],
    'volume': [1000, 1200, 1100],
    'amount': [100000, 122400, 112750]
})
```

## 模型微调

### 环境准备

#### 1. 安装Qlib
```bash
pip install pyqlib
```

#### 2. 准备Qlib数据
```bash
# 下载中国A股数据
qlib_data_path="~/.qlib/qlib_data/cn_data"
mkdir -p $qlib_data_path
cd $qlib_data_path
```

### 配置微调参数

#### 修改配置文件
编辑 `finetune/config.py`：

```python
class Config:
    def __init__(self):
        # 数据路径配置
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"
        self.dataset_path = "./data/processed_datasets"
        self.save_path = "./outputs/models"
        
        # 预训练模型路径
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-small"
        
        # 训练参数
        self.epochs = 30
        self.batch_size = 50
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5
```

### 微调流程

#### 1. 数据预处理
```bash
cd finetune
python qlib_data_preprocess.py
```

#### 2. 训练分词器
```bash
# 单GPU训练
python train_tokenizer.py

# 多GPU训练
torchrun --standalone --nproc_per_node=2 train_tokenizer.py
```

#### 3. 训练预测器
```bash
# 单GPU训练
python train_predictor.py

# 多GPU训练
torchrun --standalone --nproc_per_node=2 train_predictor.py
```

## 回测评估

### 运行回测
```bash
cd finetune
python qlib_test.py --device cuda:0
```

### 回测结果分析

#### 性能指标
- **累计收益率**：策略vs基准
- **夏普比率**：风险调整后收益
- **最大回撤**：最大损失幅度
- **胜率**：盈利交易比例

## 常见问题解决

### 1. 内存不足
**问题**：训练时出现CUDA out of memory
**解决方案**：
```python
# 减少批次大小
self.batch_size = 25

# 减少上下文长度
self.max_context = 256

# 启用梯度累积
self.accumulation_steps = 2
```

### 2. 模型加载失败
**问题**：无法从Hugging Face Hub加载模型
**解决方案**：
```python
# 检查网络连接
import requests
response = requests.get("https://huggingface.co")

# 使用本地模型路径
tokenizer = KronosTokenizer.from_pretrained("./local_model_path")
model = Kronos.from_pretrained("./local_model_path")
```

### 3. 数据格式错误
**问题**：数据列名不匹配
**解决方案**：
```python
# 检查数据列名
print("数据列名:", df.columns.tolist())

# 重命名列
df = df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})
```

## 联系和支持

### 官方资源
- **GitHub仓库**：https://github.com/shiyu-coder/Kronos
- **论文链接**：https://arxiv.org/abs/2508.02739
- **在线演示**：https://shiyu-coder.github.io/Kronos-demo/

### 引用格式
```bibtex
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets}, 
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739}, 
}
``` 