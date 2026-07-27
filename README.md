# MSSG-Net：基于多尺度序列特征与地层约束图神经网络的岩性智能识别

岩性识别同时受到测井曲线纵向变化、邻井空间关系和地层背景的影响。现有测井岩性智能识别方法未能兼顾垂向测井序列特征、横向邻井空间关联以及地层背景信息的问题，本项目构建MSSG-Net岩性识别方法。

首先，对原始测井数据进行缺失信息标记、缺失值处理、对数变换、井内标准化和衍生响应特征构建，形成统一的测井输入特征。

随后，以目标深度采样点为中心建立局部测井序列窗口，通过多尺度一维卷积提取局部响应变化、中等范围层段形态和较大范围地层趋势，并利用注意力池化获得目标深度点的纵向序列特征。

在空间特征建模阶段，以地层单元为约束，将同一口井、同一地层范围内连续且测井响应相近的采样点聚合为超节点。结合超节点的地层属性、空间位置和测井响应统计特征，建立邻井候选关系；通过动态边权学习和邻域聚合提取地层约束下的跨井空间特征。

最后，融合单井纵向序列特征与邻井空间特征，经过分类层输出岩性识别结果，并使用 Accuracy、Precision、Recall、F1-score 和 Penalty 对模型进行评价。

## 项目结构

```text
.
|-- train.py
|-- src/mssgnet/
|   |-- config.py
|   |-- preprocessing.py
|   |-- dataset.py
|   |-- model.py
|   |-- training.py
|   `-- cli.py
|-- data/README.md
|-- requirements.txt
`-- pyproject.toml
```

## 运行方式

安装 Python 3.9 或更高版本，并安装项目依赖：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

查看运行参数：

```bash
python train.py --help
```

运行训练：

```bash
python train.py \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --penalty-matrix data/penalty_matrix.npy \
  --output-dir outputs/default
```
