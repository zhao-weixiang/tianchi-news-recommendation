# 致谢

本项目修改自 https://github.com/John-Chen92/tianchi-news-recommendation 
感谢原作者的杰出工作。

#### 与原项目的差异

- 新增了 swing 召回和 twotower （双塔）召回。原因：原版 Binetwork 这路更像弱化版共现 ItemCF，作为独立召回通道说服力不够，放在简历上容易被面试官质疑。指标上 Swing 更稳，但 twotower 更有技术深度。 该项目目前选择 twotower 召回，itemcf 召回和基于 word2vec 的 i2i 召回。
- 修改了 test.sh，能正确输出日志，同时将原来的 Binetwork 召回改为 twotower 召回。

## 项目背景
零基础入门推荐系统 - 新闻推荐 Top2  

比赛地址: https://tianchi.aliyun.com/competition/entrance/531842/introduction

该比赛目前已停止提交

## 项目方案

采用3种召回方式：itemcf 召回，twotower 召回和基于 word2vec 的 i2i 召回。合并去重并删除没有召回到真实商品的用户数据后，利用特征工程+ LGB 二分类模型进行排序。

## 复现步骤

1. Ubuntu 16.04  

```
pip install requirements.txt
cd code
bash test.sh
```

2. Windows

建议直接在 PowerShell 中逐条执行 test.sh 中的命令。

```
python data.py --mode valid --logfile data.log
python recall_itemcf.py --mode valid --logfile itemcf.log
python recall_binetworkGpu.py --mode valid --logfile binetwork.log
python recall_w2v.py --mode valid --logfile w2v.log
python recall.py --mode valid --logfile recall.log
python rank_feature.py --mode valid --logfile feature.log
python rank_lgb.py --mode valid --logfile rank.log
```

## 注意

在运行 data.py 时必须把 python 版本改成3.7，运行其他 py 时建议使用python 3.11的 conda 环境

未使用的召回方案是因为效果不好，可以自己尝试和调整。
