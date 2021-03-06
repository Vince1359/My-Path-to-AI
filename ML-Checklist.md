# 机器学习项目开发自检清单

## 开发步骤

1. 落实要解决的问题。
2. 获取数据。
3. 对数据进行探索，获取灵感。
4. 对数据进行预处理，使之更好地将底层数据模式暴露给机器学习算法。
5. 探索多个不同的模型，并将效果最好的3-5个模型列出来。
6. 对模型进行调参，得到最终的模型。
7. 展示并分享解决方案。
8. 部署，监控，维护该系统。

## 落实要解决的问题

- [ ] 定义要解决的问题
- [ ] 落实解决方案的使用方式（完整系统、更大的系统中的一环）
- [ ] 如有，参考现行的解决方案，获取灵感
- [ ] 解决该问题使用的方式（监督学习/非监督学习、离线学习/在线学习等等）
- [ ] 定义解决方案的性能指标
- [ ] 判断性能指标是否与系统目标相吻合
- [ ] 落实要解决问题所要达到的最低性能
- [ ] 如有相似的问题，通过比较获得经验，或使用现成的工具
- [ ] 是否有人类专家的经验可以帮忙
- [ ] 如何用非机器学习算法解决这个问题
- [ ] 列举到目前为止所有对解决问题的假设
- [ ] 对所有假设进行一定程度的证明

## 获取数据

- 尽可能地实现自动化，从而方便获取更多的数据
- [ ] 列出需要的数据和数据的规模
- [ ] 寻找与记录获取数据的方式与地点
- [ ] 计算数据所需要占用的空间
- [ ] 检查获得与使用数据所需要的权限
- [ ] 获得所有的数据权限
- [ ] 创建工作目录与结构
- [ ] 获得数据
- [ ] 将数据转化为容易操作的格式（不改变数据内容本身）
- [ ] 对敏感信息进行保护
- [ ] 检查数据的类型和大小
- [ ] 从数据中取样出一个测试集，不要去动测试集里面的数据

## 对数据进行探索

- 尝试从与问题领域的专家交流中获得灵感
- [ ] 复制一份数据用于探索，如果数据量太大，使用合理的抽样的方法获取一个子集
- [ ] 使用Jupyter去探索数据，方便记录探索过程
- [ ] 学习每一个特征的特点：
  - [ ] 名字  
  - [ ] 类型（类别类型，整形/浮点型，有范围的数据/无范围的数据，字符串类型，对象类型等）
  - [ ] 缺失值的数量  
  - [ ] 是否有噪声以及噪声类型（随机噪声，离群值，舍入误差等）
  - [ ] 是否有可能对解决问题有用
  - [ ] 分布类型（高斯分布，均匀分布，对数分布等)
- [ ] 对监督学习任务，确定目标特征。
- [ ] 数据可视化
- [ ] 研究不同特征之间的相关性
- [ ] 研究手动解决该问题的方法
- [ ] 寻找有效的数据变换
- [ ] 落实是否还能获得对系统有用的数据
- [ ] 记录下在探索数据的过程中学到的内容

## 数据预处理

- 在复制的数据中进行操作
- 将所有对数据进行变换的代码封装在函数中（有以下好处）：
  - 当获取全新数据的时候可以很方便的实现相同的数据变换
  - 数据变换可能在不同的项目中复用  
  - 使用同样的数据变换对测试集进行相同的处理
  - 对新的样本实现相同的数据变换与处理
  - 可以将数据预处理步骤作为模型的超参数进行调整
- [ ] 数据清洗：
  - [ ] 修复或移除离群值
  - [ ] 填充空缺值（用0，平均值或中位数进行填充）或删除空缺（删除有空缺值的样本或有空缺值的特征）
- [ ] 特征选择：
  - [ ] 将对任务没有明显用处的特征去掉
- [ ] 特征工程：
  - [ ] 对连续值离散化处理（分区间）
  - [ ] 特征分解（类别特征分解，日期分解为年月日等）
  - [ ] 添加可能有效的特征变换（log( ), sqrt( ),  ^2等）
  - [ ] 将若干特征组合为新的特征
- [ ] 特征归一化：
  - [ ] 标准化（standardize）
  - [ ] 规范化（normalize）

## 探索并列举效果好的模型

- 如果数据量非常的大，可以抽样出多个小的训练数据集在可接受的时间范围内去训练多个不同的模型（神经网络或随机森林这样的复杂的模型可能需要用更多的数据）
- 尽可能自动化
- [ ] 使用默认参数快速训练多个不同种类的模型（线性回归，朴素贝叶斯，支持向量机，随机森林，神经网络）
- [ ] 比较上述模型的表现：
  - [ ] 对每个模型使用N折交叉验证，计算模型性能的均值和标准差
- [ ] 分析每个模型最重要的超参数
- [ ] 分析模型误差类型
  - [ ] 非机器学习方法会使用什么数据来避免这样的误差
- [ ] 进一步进行特征选择与特征工程
- [ ] 再对述步骤1-5实行一到两次迭代
- [ ] 列举出性能最好的3个到5个模型，最好可以包含误差类型不同的模型

## 模型调参

- 使用尽可能多的数据
- 尽可能自动化
- [ ] 使用交叉验证对模型超参数进行调优
  - [ ] 将数据变换的选择作为超参数（例如缺失值填充使用0，中位数还是平均数，还是直接丢弃特征或者丢弃样本）
  - [ ] 使用随机搜索寻找超参数，除非超参数的组合非常少，则使用网格搜索。如果训练的时间非常长，还可以考虑一下贝叶斯优化
- [ ] 尝试集成学习，将运行效果最好的几个模型整合起来
- [ ] 在测试集中对最终模型进行测试与评估
  
## 解决方案展示

- [ ] 完善文档
- [ ] 创建一个优秀的展示
- [ ] 分享解决方案如何达到目标
- [ ] 分享在解决问题的途中遇到的有趣的点
  - [ ]  描述一下有效和无效的方法
  - [ ] 列出我的假设和系统的局限性
- [ ] 对解决问题的核心进行可视化和清晰的描述

## 部署，监控，维护

- [ ] 为最后的部署做准备（使用真实的数据输入，如果是更大系统的一环，打通数据流，编写单元测试等）
- [ ] 编写监控代码，监控系统的实时表现，并在系统出现性能衰退是提供警告
  - [ ] 应对缓慢的衰退（当数据发生变化的时候模型很可能性能会下降）
  - [ ] 建立相应的人为监控方案，监控系统性能
  - [ ] 监控输入数据的性能，提前发现无效数据与错误数据（对在线学习算法尤为重要）
- [ ] 定期使用最新的数据对模型进行重新的训练（自动化）