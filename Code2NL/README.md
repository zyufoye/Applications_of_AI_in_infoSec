# 代码文档生成
代码文档生成任务基于CodeBert，在6种不同的编程语言上实现了代码文档生成任务，基于 CodeSearchNet 数据集。

## 1.依赖安装

pip install torch==1.4.0  
pip install transformers==2.5.0  

## 2.Preprocessing 数据预处理

 CodeSearchNet 数据集清理流程：  

1. 删除代码中的注释；
2. 删除代码无法解析为抽象语法树的示例；
3. 删除文档标记数小于 3 或大于 256 的示例；
4. 删除文档包含特殊标记的示例,例如<img ...>或 https:...；
5. 删除文档不是英文的示例;

清洗后的数据集可直接下载：  
https://drive.google.com/file/d/1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h/view  

## 3.Fine-tune 微调

```bash
cd code2nl

lang=php #programming language
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../data/code2nl/CodeSearchNet
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps 
```

## 4.Inference and Evaluation  推理与评估

模型微调后，推理与评估结果命令如下：
```bash
lang=php #programming language
beam_size=10
batch_size=128
source_length=256
target_length=128
output_dir=model/$lang
data_dir=../data/code2nl/CodeSearchNet
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
```

CodeSearchNet结果记录。



```bash

```
```bash

```