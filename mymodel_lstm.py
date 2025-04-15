import torch
import torch.nn as nn
from torch.autograd import Variable


# 模型构建一个LSTM层+一个Linear层
class MyLstmModel(nn.Module):
    # 定义模型
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(MyLstmModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 损失函数
        self.loss = nn.CrossEntropyLoss()
        # self.loss = torch.nn.NLLLoss()

        #     LSTM_Args:
        #     input_size: 输入张量 x 中期望的特征数量。
        #     hidden_size: 隐藏状态 h 中的特征数量。
        #     num_layers: 循环层的数量。例如，设置num_layers=2将意味着将两个LSTM堆叠在一起形成一个“堆叠的LSTM”，第二个LSTM接收第一个LSTM的输出并计算最终结果。默认值为1。
        #     bias: 如果为False，则该层不使用偏置权重 b_ih 和 b_hh。默认值为True
        #     batch_first: 如果为True，则输入和输出张量以(batch, seq, feature)的形式提供，而不是(seq, batch, feature)的形式。请注意，这不适用于隐藏状态或单元状态
        #     dropout: 如果非零，则在每个LSTM层的输出上引入一个Dropout层，除了最后一层，其丢弃概率等于dropout。默认值为0
        #     bidirectional: 如果为True，则成为一个双向LSTM。默认值为False。
        #     proj_size:如果> 0，将使用投影大小的LSTM。默认值为0。
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            batch_first=False, num_layers=self.num_layers,
                            dropout=0.4, bidirectional=False)

        # linear层:[hidden_dim,vocab_size]
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, data_input, h_0=None, c_0=None):
        # h_0 是隐藏状态，表示 LSTM 单元的记忆或潜在状态。
        # c_0 是细胞状态，用于存储长期信息。

        seq_len, batch_size = data_input.size()

        # h_0 :tensor [num_Layers * num_directions,batch,hidden_num]
        # c_0 :tensor [num_Layers * num_directions,batch,hidden_num]
        if h_0 is None or c_0 is None:
            # h_0 = torch.tensor(np.zeros((self.num_layers, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            # c_0 = torch.tensor(np.zeros((self.num_layers, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            h_0 = data_input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = data_input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)

        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)

        # xs_embedding.shape: (seq, batch, hidden_dim)
        xs_embedding = self.embeddings(data_input)

        pre, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))
        # hidden_drop = self.dropout(hidden)
        # flatten_hidden = self.flatten(hidden_drop)
        # pre = self.linear(flatten_hidden)

        # ((seq_len * batch_size),hidden_dim)
        pre = self.linear(pre.view(seq_len*batch_size, -1))
        return pre, (h_0, c_0)
