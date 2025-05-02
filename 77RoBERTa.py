import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 配置类, 用于定义动态掩码的参数
class DynamicMaskConfig:
    def __init__(self, mask_token_id, mask_probability=0.15, replace_with_mask=0.8, replace_with_random=0.1):
        """
        动态掩码配置类。
        :param mask_token_id: 掩码标记的 ID
        :param mask_probability: 掩码的概率，默认为 0.15
        :param replace_with_mask: 使用掩码标记替换的概率，默认为 0.8
        :param replace_with_random: 使用随机标记替换的概率，默认为 0.1
        """
        self.mask_token_id = mask_token_id
        self.mask_probability = mask_probability
        self.replace_with_mask = replace_with_mask
        self.replace_with_random = replace_with_random


# RoBERTa的嵌入层
class RobertaEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, layer_norm_eps, pad_token_id):
        """
        RoBERTa 嵌入层。
        :param vocab_size: 词汇表大小
        :param hidden_size: 隐藏层大小
        :param max_position_embeddings: 最大位置嵌入
        :param type_vocab_size: 类型词汇表大小
        :param layer_norm_eps: LayerNorm 的 epsilon 值
        :param pad_token_id: 填充标记的 ID
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)  # 单词嵌入
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)  # 位置嵌入
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)  # 类型嵌入
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.1)  # Dropout

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        前向传播。
        :param input_ids: 输入的标记 ID
        :param token_type_ids: 类型标记 ID，默认为 None
        :param position_ids: 位置标记 ID，默认为 None
        :return: 嵌入后的输出
        """
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# RoBERTa的自注意力层
class RobertaSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        """
        RoBERTa 自注意力层。
        :param hidden_size: 隐藏层大小
        :param num_attention_heads: 注意力头的数量
        :param attention_probs_dropout_prob: 注意力概率的 Dropout 概率
        """
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)  # 查询线性层
        self.key = nn.Linear(hidden_size, self.all_head_size)  # 键线性层
        self.value = nn.Linear(hidden_size, self.all_head_size)  # 值线性层

        self.dropout = nn.Dropout(attention_probs_dropout_prob)  # Dropout

    def transpose_for_scores(self, x):
        """
        调整张量形状以计算分数。
        :param x: 输入张量
        :return: 调整后的张量
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播。
        :param hidden_states: 隐藏状态
        :param attention_mask: 注意力掩码，默认为 None
        :return: 自注意力的输出
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


# RoBERTa的自注意力输出层
class RobertaSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps):
        """
        RoBERTa 自注意力输出层。
        :param hidden_size: 隐藏层大小
        :param layer_norm_eps: LayerNorm 的 epsilon 值
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)  # 线性层
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)  # LayerNorm
        self.dropout = nn.Dropout(0.1)  # Dropout

    def forward(self, hidden_states, input_tensor):
        """
        前向传播。
        :param hidden_states: 隐藏状态
        :param input_tensor: 输入张量
        :return: 输出张量
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# RoBERTa的注意力层
class RobertaAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, layer_norm_eps):
        """
        RoBERTa 注意力层。
        :param hidden_size: 隐藏层大小
        :param num_attention_heads: 注意力头的数量
        :param attention_probs_dropout_prob: 注意力概率的 Dropout 概率
        :param layer_norm_eps: LayerNorm 的 epsilon 值
        """
        super().__init__()
        self.self = RobertaSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = RobertaSelfOutput(hidden_size, layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播。
        :param hidden_states: 隐藏状态
        :param attention_mask: 注意力掩码，默认为 None
        :return: 注意力的输出
        """
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


# RoBERTa的中间层
class RobertaIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        RoBERTa 中间层。
        :param hidden_size: 隐藏层大小
        :param intermediate_size: 中间层大小
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)  # 线性层
        self.intermediate_act_fn = F.gelu  # 激活函数

    def forward(self, hidden_states):
        """
        前向传播。
        :param hidden_states: 隐藏状态
        :return: 输出张量
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# RoBERTa的输出层
class RobertaOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, layer_norm_eps):
        """
        RoBERTa 输出层。
        :param intermediate_size: 中间层大小
        :param hidden_size: 隐藏层大小
        :param layer_norm_eps: LayerNorm 的 epsilon 值
        """
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)  # 线性层
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)  # LayerNorm
        self.dropout = nn.Dropout(0.1)  # Dropout

    def forward(self, hidden_states, input_tensor):
        """
        前向传播。
        :param hidden_states: 隐藏状态
        :param input_tensor: 输入张量
        :return: 输出张量
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# RoBERTa的单层
class RobertaLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob,
                 layer_norm_eps):
        """
        RoBERTa 单层。
        :param hidden_size: 隐藏层大小
        :param num_attention_heads: 注意力头的数量
        :param intermediate_size: 中间层大小
        :param attention_probs_dropout_prob: 注意力概率的 Dropout 概率
        :param layer_norm_eps: LayerNorm 的 epsilon 值
        """
        super().__init__()
        self.attention = RobertaAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob,
                                          layer_norm_eps)
        self.intermediate = RobertaIntermediate(hidden_size, intermediate_size)
        self.output = RobertaOutput(intermediate_size, hidden_size, layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播。
        :param hidden_states: 隐藏状态
        :param attention_mask: 注意力掩码，默认为 None
        :return: 输出张量
        """
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# RoBERTa的编码器
class RobertaEncoder(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, num_hidden_layers,
                 attention_probs_dropout_prob, layer_norm_eps):
        """
        RoBERTa 编码器。
        :param hidden_size: 隐藏层大小
        :param num_attention_heads: 注意力头的数量
        :param intermediate_size: 中间层大小
        :param num_hidden_layers: 隐藏层的数量
        :param attention_probs_dropout_prob: 注意力概率的 Dropout 概率
        :param layer_norm_eps: LayerNorm 的 epsilon 值
        """
        super().__init__()
        self.layer = nn.ModuleList([RobertaLayer(hidden_size, num_attention_heads, intermediate_size,
                                                 attention_probs_dropout_prob, layer_norm_eps) for _ in
                                    range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播。
        :param hidden_states: 隐藏状态
        :param attention_mask: 注意力掩码，默认为 None
        :return: 所有隐藏层的输出
        """
        all_hidden_states = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)
        return all_hidden_states


# RoBERTa的池化器
class RobertaPooler(nn.Module):
    def __init__(self, hidden_size):
        """
        RoBERTa 池化器。
        :param hidden_size: 隐藏层大小
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)  # 线性层
        self.activation = nn.Tanh()  # 激活函数

    def forward(self, hidden_states):
        """
        前向传播。
        :param hidden_states: 隐藏状态
        :return: 池化后的输出
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# RoBERTa的语言模型头
class RobertaLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps):
        """
        RoBERTa 语言模型头。
        :param hidden_size: 隐藏层大小
        :param vocab_size: 词汇表大小
        :param layer_norm_eps: LayerNorm 的 epsilon 值
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)  # 线性层
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)  # LayerNorm

        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)  # 解码器
        self.bias = nn.Parameter(torch.zeros(vocab_size))  # 偏置

    def forward(self, features, **kwargs):
        """
        前向传播。
        :param features: 输入特征
        :return: 输出张量
        """
        x = self.dense(features)
        x = self.layer_norm(x)

        x = self.decoder(x) + self.bias
        return x


# RoBERTa 的预训练模型基类
class RobertaPreTrainedModel(nn.Module):
    def __init__(self):
        """
        RoBERTa 预训练模型基类。
        """
        super().__init__()

    def _init_weights(self, module):
        """
        初始化权重。
        :param module: 模块
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def apply(self, function):
        """
        应用函数到所有子模块。
        :param function: 函数
        """
        for module in self.children():
            module.apply(function)
        function(self)


# RoBERTa模型
class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config):
        """
        RoBERTa 模型。
        :param config: 配置
        """
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            max_position_embeddings=config["max_position_embeddings"],
            type_vocab_size=config["type_vocab_size"],
            layer_norm_eps=config["layer_norm_eps"],
            pad_token_id=config["pad_token_id"]
        )
        self.encoder = RobertaEncoder(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
            layer_norm_eps=config["layer_norm_eps"]
        )
        self.pooler = RobertaPooler(hidden_size=config["hidden_size"])
        self.apply(self._init_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        """
        前向传播。
        :param input_ids: 输入的标记 ID
        :param attention_mask: 注意力掩码，默认为 None
        :param token_type_ids: 类型标记 ID，默认为 None
        :param position_ids: 位置标记 ID，默认为 None
        :return: 序列输出和池化输出
        """
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_ids.size())

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        sequence_output = encoder_outputs[-1]
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output

    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        获取扩展的注意力掩码。
        :param attention_mask: 注意力掩码
        :param input_shape: 输入形状
        :return: 扩展的注意力掩码
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            ))

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


# RoBERTa的掩码语言模型
class RobertaForMaskedLM(RobertaPreTrainedModel):
    def __init__(self, config, mask_config):
        """
        RoBERTa 的掩码语言模型。
        :param config: 配置
        :param mask_config: 掩码配置
        """
        super().__init__()
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(
            hidden_size=config["hidden_size"],
            vocab_size=config["vocab_size"],
            layer_norm_eps=config["layer_norm_eps"]
        )
        self.config = config
        self.mask_config = mask_config
        self.apply(self._init_weights)

    def apply_mask(self, input_ids):
        """
        应用掩码。
        :param input_ids: 输入的标记 ID
        :return: 掩码后的输入和标签
        """
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_config.mask_probability, device=input_ids.device)

        special_tokens_mask = (input_ids == self.roberta.embeddings.word_embeddings.padding_idx)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        if self.mask_config.replace_with_mask:
            mask_token_indices = torch.bernoulli(
                torch.full(labels.shape, self.mask_config.replace_with_mask, device=input_ids.device)
            ).bool() & masked_indices
            input_ids[mask_token_indices] = self.mask_config.mask_token_id

        if self.mask_config.replace_with_random:
            random_token_indices = torch.bernoulli(
                torch.full(labels.shape, self.mask_config.replace_with_random, device=input_ids.device)
            ).bool() & masked_indices & ~mask_token_indices
            random_tokens = torch.randint(
                0,
                self.roberta.embeddings.word_embeddings.num_embeddings,
                labels.shape,
                dtype=torch.long,
                device=input_ids.device
            )
            input_ids[random_token_indices] = random_tokens[random_token_indices]

        return input_ids, labels

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None
    ):
        """
        前向传播。
        :param input_ids: 输入的标记 ID
        :param attention_mask: 注意力掩码，默认为 None
        :param token_type_ids: 类型标记 ID，默认为 None
        :param position_ids: 位置标记 ID，默认为 None
        :param labels: 标签，默认为 None
        :return: 损失、预测分数、隐藏状态和注意力权重
        """
        if labels is None:
            input_ids, labels = self.apply_mask(input_ids)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config["vocab_size"]), labels.view(-1))

        return {"loss": masked_lm_loss, "logits": prediction_scores, "hidden_states": outputs[0:], "attentions": []}

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, labels=None):
        """
        调用模型。
        :param input_ids: 输入的标记 ID
        :param attention_mask: 注意力掩码，默认为 None
        :param token_type_ids: 类型标记 ID，默认为 None
        :param position_ids: 位置标记 ID，默认为 None
        :param labels: 标签，默认为 None
        :return: 损失、预测分数、隐藏状态和注意力权重
        """
        return self.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels
        )


if __name__ == "__main__":
    config = {
        "vocab_size": 50265,
        "hidden_size": 768,
        "max_position_embeddings": 512,
        "type_vocab_size": 1,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 1,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "attention_probs_dropout_prob": 0.1
    }
    mask_config = DynamicMaskConfig(mask_token_id=50264)

    model = RobertaForMaskedLM(config, mask_config)

    input_ids = torch.randint(2, config["vocab_size"], (32, 128))  # 确保不包含 padding token
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

    outputs = model(input_ids, attention_mask=attention_mask)
    print(outputs["loss"])
    print(outputs["logits"].shape)
