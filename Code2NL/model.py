import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy

# Seq2Seq 模型核心架构
# Seq2Seq 继承自 nn.Module，作用是把一个encoder和一个decoder组装成一个完整的序列到序列模型
# 类似于“输入代码 → 输出自然语言描述”
# encoder：通常是 RobertaModel（即 CodeBERT 本体，理解输入代码）
# decoder：是一个 Transformer 解码器，用来逐步生成目标文本（自然语言描述）
# beam search 参数：在推理时用束搜索（beam search）提高生成质量

# 初始化时传入编码器、解码器、配置、以及集束搜索需要的参数
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder # 编码器，用于处理输入序列，代码token
        self.decoder=decoder # 标准 Transformer decoder，基于编码结果逐步生成目标句子
        self.config=config  # 包含模型参数，如隐藏层维度 hidden_size、词表大小 vocab_size
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048))) # 掩码矩阵，下三角矩阵
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 对编码后的隐藏表示做一次线性变换
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 语言模型头，把隐藏状态映射到词表大小（得到对每个词的 logits）
        self.lsm = nn.LogSoftmax(dim=-1)  # 对 logits 做 log softmax，得到预测分布
        self.tie_weights()# 把解码器的embedding权重和lm_head的权重绑在一起（共享参数），减少模型大小，同时提升效果
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)     
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None):   
        # 编码器前向 
        # source_ids: 输入代码的 token ids
        # source_mask: 输入代码的 attention mask
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()

        # 仅在训练时，进入训练/监督解码路径
        if target_ids is not None:  
            # 自回归注意力 mask，下三角矩阵
            # self.bias 是初始化时注册的 下三角矩阵（lower-triangular）大小 2048×2048，取前 S_tgt 子块得到 [S_tgt, S_tgt]
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs