import torch
import tqdm as tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import train_vec
from mymodel_lstm import MyLstmModel
from torchnet import meter

from mymodel_rnn import MyRnnModel

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device: ", device)

# 数据预处理的过程都是一样的
def get_data_args(batch_size):
    # train_vec->数据预处理：Json->txt，拆分，词向量
    data, (w1, word_2_index, index_2_word) = train_vec()

    # w1:输入层到隐层的权重矩阵
    # [vocab_size, embedding_dim]
    # vocab_size :词汇表"词库"的大小 = word_size
    # embedding_dim :词嵌入的维度 = vector_size = embedding_num
    vocab_size, embedding_dim = w1.shape

    # DataLoader批量加载数据 -> 数据在 DataLoader 对象中已经被封装好了
    # batch_size: 每个批次（batch）中包含的样本数量 -> 将batch_size个数据打包成一份
    # shuffle: 是否在每个 epoch 之前打乱数据集的顺序 ;False->不打乱
    # num_workers :数据加载的子进程数量(默认为0)
    # pin_memory : 是否将数据加载到 CUDA 的固定内(这里是CPU版的pytorch)
    # drop_last : 是否丢弃最后一个不完整的批次->确保每个批次都有相同数量的样本
    dataloader = DataLoader(data, batch_size, shuffle=True, drop_last=True)

    # shuffle=True则需要确保 __getitem__ 方法返回的数据长度是固定的，所有序列的长度要求一致
    return embedding_dim, vocab_size, index_2_word, dataloader


# 获取参数 在这里修改
def get_args():
    # (*)
    batch_size = 64
    # (*)hidden_num:
    hidden_dim = 130
    # (*)lr : 学习率
    lr = 1e-3
    # (*)训练过程中的轮数
    epochs = 40
    layers_num = 2

    return batch_size, hidden_dim, layers_num, lr, epochs


def train(type):
    batch_size, hidden_dim, layers_num, lr, epochs = get_args()
    # return: embedding_dim, vocab_size, index_2_word, dataloader
    embedding_dim, vocab_size, index_2_word, dataloader = get_data_args(batch_size)

    # 默认 -> 输入不存在的数字也训练的是使用LSTM层的网络
    model_file = './model/model_lstm'
    model = MyLstmModel(vocab_size, embedding_dim, hidden_dim, layers_num)

    if type == 3:
        return

    elif type == 1:
        model_file = './model/model_lstm'
        # MyLstmModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        model = MyLstmModel(vocab_size, embedding_dim, hidden_dim, layers_num)

    elif type == 2:
        model_file = './model/model_rnn'
        model = MyRnnModel(vocab_size, embedding_dim, hidden_dim, layers_num)

    model = model.to(device)
    # 损失函数
    criterion = model.loss

    # (*)优化器，梯度更新
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 计算每个epoch的损失值
    loss_meter = meter.AverageValueMeter()
    period = []
    loss2 = []

    for e in range(epochs):
        loss_meter.reset()

        # sequence_length(一首古诗) : 32
        # xs_embedding.shape: ([batch_size,sequence_length,embedding_dim])
        # ys_index.shape: ([batch_size,sequence_length])
        # 迭代 DataLoader 对象时，才会调用 MyDataset 中的 __getitem__
        for batch_index, data in tqdm.tqdm(enumerate(dataloader)):

            data = data.long().transpose(0, 1).contiguous()
            data = data.to(device)

            optimizer.zero_grad()
            # target 为真值
            x_train, y_train = Variable(data[:-1, :]), Variable(data[1:, :])

            pre, _ = model(x_train)
            loss = criterion(pre, y_train.view(-1))
            loss.backward()

            # nn.utils.clip_grad_norm(model.parameters(), 1)

            optimizer.step()

            # 交叉函数损失
            loss_meter.add(loss.item())
            period.append(batch_index + e * len(dataloader))
            loss2.append(loss_meter.value()[0])

            # 定期打印，观察
            if batch_index % 20 == 0:
                print(loss)
                # 每20个epoch保存一次model
                torch.save(model.state_dict(), model_file)

    # 保存最终model
    torch.save(model.state_dict(), model_file)
