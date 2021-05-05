
# 概述
1、论文复现《[Named Entity Recognition as Dependency Parsing](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)》。使用双仿射注意力机制Biaffine的思想来进行命名实体识别。<br>
2、模型的具体思路描述可以见[知乎](https://zhuanlan.zhihu.com/p/369851456)。<br>
3、训练数据来自于MSRA命名实体识别训练语料，已经做相应的修改转换。<br>

# 环境要求
```
pytorch >=1.6.0
transformers>=3.4.0
```
# 运行步骤
1、去huggingface[官网](https://huggingface.co/models)下载BERT预训练权重，然后并放在`./pretrained_model/`文件夹下<br>
2、在`./utils/arguments_parse.py`中修改BERT预训练模型的路径<br>
3、运行`train.py`进行训练<br>
4、运行`predict.py`进行预测

# 项目结构
```
│  eval_func.py  
│  LICENSE
│  predict.py            #预测脚本
│  README.md     
│  train.py              #训练脚本
│
├─checkpoints
├─data                   #数据
│      test.json         
│      train.json
│
├─data_preprocessing
│      data_prepro.py
│      tools.py
│
├─log
├─model
│  │  model.py            #模型结构
│  │
│  ├─loss_function        #损失函数
│  │      binary_cross_entropy.py
│  │      cross_entropy_loss.py
│  │      focal_loss.py
│  │      multilabel_cross_entropy.py
│  │      span_loss.py
│  │
│  └─metrics               #测评函数
│          metrics.py
│
├─output                   #输出文件，预测得到的结果放在这里
│      result.json
│
├─pretrained_model         #改文件夹用来存放预训练模型
│  └─chinese_roberta_wwm_ext
└─utils
        arguments_parse.py  #设置模型训练相关的参数
        logger.py           #存放日志等脚本
```

