# Kronos 项目学习文档

## 项目概述

**Kronos** 是一个专门为金融市场K线序列设计的开源基础模型，它是第一个针对金融K线（K-lines）的开源基础模型，在超过45个全球交易所的数据上进行了预训练。

### 核心特点
- **专业化设计**：专门处理金融数据的高噪声特性
- **两阶段框架**：专用分词器 + 自回归Transformer
- **多维度支持**：支持OHLCV（开高低收量）数据
- **开源可用**：提供多个规模的预训练模型

## 项目架构

### 整体架构图
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   原始K线数据    │ -> │   专用分词器     │ -> │   离散化token   │
│   (OHLCV)      │    │  KronosTokenizer│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   自回归Transformer│
                       │   Kronos Model  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   预测结果      │
                       │   (未来K线)     │
                       └─────────────────┘
```

### 两阶段框架详解

#### 第一阶段：专用分词器
- **功能**：将连续的、多维的K线数据量化为分层离散token
- **技术**：使用Binary Spherical Quantization (BSQuantizer)
- **优势**：能够处理金融数据的特殊分布特性

#### 第二阶段：自回归Transformer
- **功能**：在量化后的token上进行预训练
- **应用**：可作为多种量化任务的统一模型
- **架构**：基于Transformer的decoder-only架构

## 文件结构详解

### 核心模型文件

#### `model/kronos.py`
**功能**：Kronos模型的核心实现文件
**主要类**：
- `KronosTokenizer`：专用分词器类
  - 使用混合量化方法
  - 结合编码器和解码器Transformer块
  - 实现Binary Spherical Quantization (BSQuantizer)
- `Kronos`：主要的Kronos模型类
  - 继承自PyTorchModelHubMixin
  - 支持从Hugging Face Hub加载预训练模型
- `KronosPredictor`：预测器类
  - 处理数据预处理、归一化、预测和反归一化
  - 提供从原始数据到预测结果的端到端解决方案

**关键参数**：
- `d_in`：输入维度
- `d_model`：模型维度
- `n_heads`：注意力头数
- `ff_dim`：前馈网络维度
- `n_enc_layers`：编码器层数
- `n_dec_layers`：解码器层数

#### `model/module.py`
**功能**：包含模型的基础组件和工具类
**主要组件**：
- `BinarySphericalQuantizer`：二进制球形量化器
  - 实现论文中的量化方法
  - 支持可微分熵计算
  - 提供分组量化和NCE损失计算
- `TransformerBlock`：Transformer块实现
- `DifferentiableEntropyFunction`：可微分熵函数
- `codebook_entropy`：码本熵计算函数

**技术特点**：
- 支持软熵和硬熵计算
- 实现分组量化以提高效率
- 提供多种熵计算策略

### 示例文件

#### `examples/prediction_example.py`
**功能**：完整的预测示例脚本
**主要功能**：
- 模型和分词器加载
- 数据预处理和准备
- 预测执行
- 结果可视化

**使用流程**：
1. 加载预训练的Kronos模型和分词器
2. 实例化预测器
3. 准备输入数据（历史K线数据）
4. 执行预测
5. 可视化预测结果

#### `examples/prediction_wo_vol_example.py`
**功能**：不包含成交量数据的预测示例
**适用场景**：当数据中缺少成交量信息时使用

### 微调相关文件

#### `finetune/config.py`
**功能**：微调配置管理
**主要配置项**：
- **数据参数**：Qlib数据路径、时间范围、特征列表
- **训练参数**：批次大小、学习率、训练轮数
- **模型路径**：预训练模型路径、保存路径
- **实验配置**：Comet ML集成、随机种子等

**关键配置**：
```python
# 数据参数
self.qlib_data_path = "~/.qlib/qlib_data/cn_data"
self.lookback_window = 90  # 回看窗口
self.predict_window = 10   # 预测窗口
self.max_context = 512     # 最大上下文长度

# 训练参数
self.epochs = 30
self.batch_size = 50
self.tokenizer_learning_rate = 2e-4
self.predictor_learning_rate = 4e-5
```

#### `finetune/dataset.py`
**功能**：数据集处理和加载
**主要功能**：
- 数据预处理和特征工程
- 训练/验证/测试集分割
- 数据加载器实现

#### `finetune/train_tokenizer.py`
**功能**：分词器微调训练脚本
**技术特点**：
- 支持多GPU训练
- 使用torchrun进行分布式训练
- 自动保存最佳检查点

#### `finetune/train_predictor.py`
**功能**：预测器微调训练脚本
**训练策略**：
- 两阶段训练：先训练分词器，再训练预测器
- 支持梯度累积
- 集成Comet ML实验跟踪

#### `finetune/qlib_data_preprocess.py`
**功能**：Qlib数据预处理脚本
**处理流程**：
1. 从Qlib加载原始市场数据
2. 数据清洗和特征工程
3. 时间序列分割
4. 保存为pickle格式

#### `finetune/qlib_test.py`
**功能**：微调模型回测评估脚本
**评估内容**：
- 模型性能测试
- 简单策略回测
- 结果可视化

### 工具文件

#### `finetune/utils/training_utils.py`
**功能**：训练相关的工具函数
**包含功能**：
- 训练循环管理
- 损失计算
- 模型保存和加载
- 性能指标计算

### 数据文件

#### `examples/data/XSHG_5min_600977.csv`
**功能**：示例数据文件
**数据格式**：包含时间戳、OHLCV数据的5分钟K线数据
**用途**：用于演示预测功能

### 配置文件

#### `requirements.txt`
**功能**：项目依赖管理
**主要依赖**：
- `torch`：PyTorch深度学习框架
- `numpy`：数值计算库
- `pandas`：数据处理库
- `huggingface_hub`：模型仓库访问
- `matplotlib`：数据可视化
- `einops`：张量操作库

## 技术架构特点

### 1. 模块化设计
- 核心模型、分词器、预测器分离
- 支持独立训练和微调
- 易于扩展和维护

### 2. 分布式训练支持
- 使用torchrun进行多GPU训练
- 支持梯度累积
- 兼容主流分布式训练框架

### 3. 模型版本管理
- 集成Hugging Face Hub
- 支持本地和远程模型加载
- 版本控制和模型共享

### 4. 实验管理
- Comet ML集成
- 训练过程监控
- 实验结果跟踪

## 应用场景

### 1. 金融预测
- 股票价格预测
- 加密货币价格预测
- 期货价格预测

### 2. 量化交易
- 策略信号生成
- 风险管理
- 投资组合优化

### 3. 研究应用
- 金融市场研究
- 时间序列分析
- 机器学习研究

## 技术优势

### 1. 专业化设计
- 针对金融数据特性优化
- 处理高噪声数据能力强
- 支持多维度特征

### 2. 开源可用
- MIT许可证
- 完整的训练和微调代码
- 活跃的社区支持

### 3. 易于使用
- 简洁的API设计
- 完整的示例代码
- 详细的文档说明

### 4. 可扩展性
- 支持自定义数据集
- 灵活的配置选项
- 模块化架构设计 