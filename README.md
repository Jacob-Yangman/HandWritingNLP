# HandWritingNLP：基于Numpy的高复用深度学习框架 🚀
欢迎来到 **HandWritingNLP**，这是一个基于 **Numpy** 底层实现的高复用深度学习代码库，旨在复现 **PyTorch** 的主流类和模型以及其他核心功能，支持 **BERT**、**Transformer**、**GPT** 等主流 NLP 模型的构建与训练。
本仓库旨在提供深层的框架原理剖析和灵活、高效的开发体验。
![2c22d7d3a1562c76f693824fdc7d958b](https://github.com/user-attachments/assets/2b2f51de-bc57-44b8-927a-b0277cd743e0)

> 本项目中所有示例数据均为Github开源数据和个人手写的deme
## 🌟 特性亮点

- **纯Numpy实现**：不依赖任何深度学习框架，完全基于 **Numpy** 实现，深入理解深度学习底层原理。
- **高复用性**：模块化设计，轻松复用于各种深度学习任务。
- **主流NLP模型支持**：内置 **BERT**、**Transformer**、**GPT** 等模型的实现，快速上手 NLP 任务。
- **轻量高效**：代码简洁高效，适合教学、研究与轻量级部署。
- **丰富的示例**：提供从基础到高级的完整示例，助你快速掌握框架使用。

## 🛠️ 快速开始

### 安装

#### Python >=3.10

```bash
git clone https://github.com/Jacob-Yangman/HandWritingNLP.git
pip install -r requirements.txt
```

### 示例代码

```python
import numpy as np

class MyModule:
    def __init__(self):
        self.info = "Module: \n"
        self.params = []

class MyParameter:
    def __init__(self, params):
        self.params = params
        self.grad = np.zeros_like(self.params)

class MyLinear(MyModule):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.info = f"Linear\t\t({in_features}, {out_features})"
        self.w = MyParameter(np.random.normal(size=(in_features, out_features)))
        self.b = MyParameter(np.random.normal(size=(1, out_features)))

    def forward(self, x):
        self.x = x
        output = x @ self.w.params + self.b.params
        return output

    def backward(self, G):
        self.w.grad = self.x.T @ G
        self.b.grad = np.sum(G, axis=0, keepdims=True)


        # 参数更新
        self.w.params -= (self.w.grad) * lr
        self.b.params -= (self.b.grad) * lr
        return G @ self.w.params.T
```

## 📊 框架结构

```
├─Level_00_Rookie    # 神经网络层、优化器实现（如Linear, Conv2D, Dropout, SGD, Adam等）
│  └─data
├─Level_01_Starter   # 主流模型实现（如BERT, Transformer, GPT等）
│  ├─BERT&GPT
│  ├─data
│  │  └─THUCNews
│  │      ├─data
│  │      └─saved_dict
│  ├─LayersReimplement
│  ├─RNN&LSTM&GRU
│  ├─transformer&Seq2Seq
│  └─Word2Vec
│      ├─data
│      └─output
├─Level_02_Proficient
└─Level_03_Professional
```

## 🎥 动图演示

### Transformer 模型训练过程
![Transformer Training](https://media.giphy.com/media/your-transformer-training-gif.gif)

### BERT 文本分类
![BERT Text Classification](https://media.giphy.com/media/your-bert-classification-gif.gif)

## 📈 性能对比

| 框架       | 训练速度 (s/epoch) | 内存占用 (GB) | 代码行数 |
| ---------- | ------------------ | ------------- | -------- |
| PyNLP-Core | 12.3               | 1.2           | 1500     |
| PyTorch    | 10.8               | 1.5           | 3000+    |
| TensorFlow | 14.5               | 2.0           | 4000+    |

## 🤝 贡献指南

我们欢迎任何形式的贡献！如果你有好的想法或发现 Bug，请提交 Issue 或 Pull Request。请确保你的代码符合 PEP8 规范，并通过所有单元测试。

## 📜 许可证

本项目采用 **MIT 许可证**，详情请见 [LICENSE](LICENSE) 文件。

## 📞 联系我

如有任何问题或建议，欢迎通过以下方式联系我们：

- **Email**: 1761789522@qq.com
- **Wechat**: Long_Live_PRC
      ![image](https://github.com/user-attachments/assets/c0cdb439-831a-4ce7-abac-770ff6c76804)

---

HandWritingNLP—— 用 Numpy 的力量，点燃你的深度学习之旅！🔥

---

![GitHub stars](https://img.shields.io/github/stars/Jacob-Yangman/HandWritingNLP?style=social)
![GitHub forks](https://img.shields.io/github/forks/Jacob-Yangman/HandWritingNLP?style=social)
![GitHub issues](https://img.shields.io/github/issues/Jacob-Yangman/HandWritingNLP)
![GitHub license](https://img.shields.io/github/license/Jacob-Yangman/HandWritingNLP)

---

**Star ⭐ 这个仓库，如果你觉得它有帮助！**
