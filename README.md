# TinyLM

> 此项目核心代码来自于优秀开源项目 [MiniMind](https://github.com/jingyaogong/minimind)，MiniMind 项目采用 Apache License 2.0 协议，向其作者致敬。

此项目将详细记录如何从头开始训练一个小型语言模型。一套完整的小模型开发标准流程：

1. 数据集准备与预处理
2. Tokenizer 训练
3. Dataset 构建
4. Model 构建
5. Pre-training 预训练
6. SFT 监督微调
7. DPO 偏好对齐
8. 部署

此项目完整目录结构如下：

```python
NanoMind/
├── data/
│   ├── pretrain_hq.jsonl
│   |── sft_data.jsonl
<<<<<<< HEAD
=======
│   ├── pretrain_hq.bin
>>>>>>> da8a0fd5f7950b005684b0a094219ef543e49ae8
│   └── preprocess.py
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── dataset/
│   ├── dataset_pretrain.py 
│   ├── dataset_sft.py
│   └── dataset_dpo.py
├── model/
│   ├── __init__.py
│   ├── configuration_tinylm.py
│   └── modeling_tinylm.py
├── scripts/
│   ├── run_pretrain.py
│   ├── run_sft.py
│   └── run_dpo.py
├── utils/
│   ├── helpers.py
│   └── loss_functions.py
├── checkpoints/
├── inference.py
├── requirements.txt
└── train.sh
```

## 第一步：数据集准备与预处理

预训练数据集采用 MiniMind 项目提供的 `pretrain_hq.jsonl` ，该数据集由 MiniMind 项目作者将**匠数科技大模型数据集**中的中文部分提取出来，并清洗出字符长度小于 512 的大约 1.6GB 的预料拼接而成的。

有监督微调阶段数据集同样采用 MiniMind 项目提供的 `sft_512.jsonl` ，该数据集同样由 MiniMind 项目作者整合自**匠数科技大模型有监督微调数据集**，并清洗出字符长度小于 512 的大约 7.5GB 的预料拼接而成的。

数据集下载地址：[https://www.modelscope.cn/datasets/gongjy/minimind_dataset]()。

将下载的数据集文件放到 `./data/` 目录下。

为了优化数据读取，降低GPU 在训练时产生明显的 I/O 等待，我们采用 **“一次性预分词 + 内存映射”** 的方案，将数据 Packing 化并持久化为二进制文件，直接运行脚本 `preprocess.py` 。

## 第二步：Tokenizer训练

在大模型训练中， **Tokenizer 并不是简单的“分词工具”** ，而是承担着一个更基础也更关键的角色：

> Tokenizer 定义了离散文本世界（字符串）与神经网络可建模的连续向量世界之间的一套唯一、稳定且可学习的映射规则。

模型在训练和推理过程中， 永远不会直接看到字符或字符串 ，而只接收由 Tokenizer 生成的  Token ID 序列 ，并在该离散符号空间上学习条件概率分布。

例如，对于原始文本：`我喜欢机器学习`。

Tokenizer 会将其转换为类似如下的整数序列：`[1532, 874, 9021, 76, 3312]`。

这些 **token_id** 才是模型在预训练与微调阶段真正建模和预测的对象。

Tokenizer 的设计会直接影响：

* 模型的输入序列长度分布
* 词表规模（vocabulary size）与 embedding 参数量
* 预测空间大小与训练难度
* 模型对低频字符、罕见符号和异常输入的鲁棒性

本项目中，Tokenizer 的训练核心流程为：

1. 从大规模预训练语料（`.jsonl` 格式）中逐行读取原始文本；
2. 基于语料的真实字符分布训练 Tokenizer；
3. 生成最终用于模型训练与推理的词表与分词规则。

具体而言，本项目采用 **ByteLevel BPE（Byte-Pair Encoding）算法** 来训练 Tokenizer。最终，Tokenizer 会生成完整的分词配置与词表文件 `tokenizer.json`，并配合 `tokenizer_config.json` 一同使用，作为后续预训练、SFT 与 DPO 阶段的统一文本编码规则。

需要强调的是：

> Tokenizer 一旦确定，就必须在整个模型生命周期中保持一致。

预训练、微调与推理阶段必须使用完全相同的 Tokenizer 配置，否则将导致 token 语义不一致，模型行为失效。

当然，也可以自行构造词表并训练新的 Tokenizer，或直接复用已有开源大模型的分词器，各有优劣。自行训练的分词器可以灵活控制词表规模与内容，但通常会带来较低的文本压缩率。为控制模型体积，本项目的词表规模设置为 6400。

> 注：在实际工程中，**绝大多数模型训练并不需要重新训练 Tokenizer**。
> 这是一个工程化决策：理解 ByteLevel BPE、分析 tokenizer 与语料分布的耦合关系是研究者应做的事情，但并非每个使用者都需要重复这一过程。因此，在本项目中，我们直接使用 MiniMind 项目作者训练好的 `tokenizer.json`。

## 第三步：Dataset 构建

## 技术说明：

1. ByteLevel BPE（Byte-Pair Encoding）算法
