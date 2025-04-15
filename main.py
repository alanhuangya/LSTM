import pickle
import random
import torch
from opencc import OpenCC
from torch.autograd import Variable
from data import train_vec
from mymodel_lstm import MyLstmModel
from mymodel_rnn import MyRnnModel
from train import get_args, train
import torch.nn.functional as f

device = "cuda" if torch.cuda.is_available() else "cpu"
vec_params_file = 'vec_params.pkl'
org_file = './txt_dataset/song.txt'

# 默认是Test
# 1: 训练lstm 2：训练rnn 3：test
mode = 3


# 加载模型
def load_model(file):
    all_data, (w1, word_2_index, index_2_word) = train_vec()
    _, hidden_dim, num_layers, _, _ = get_args()
    vocab_size, embedding_dim = w1.shape

    # 在函数内部也明确设备，增加代码清晰度
    current_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"将在 {current_device} 设备上加载模型 '{file}'")  # 添加打印信息方便调试

    if file == './model/model_lstm':
        model = MyLstmModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        # 使用 map_location 参数，这是推荐的加载模型到特定设备的方式
        # weights_only=True 是为了安全和消除警告
        model.load_state_dict(torch.load(file, map_location=current_device, weights_only=True))
        model.eval()  # 设置为评估模式
        # 再次显式移动，确保万无一失 (理论上 map_location 足够，但多一步更保险)
        model = model.to(current_device)
        print(f"模型 {file} 已移至 {next(model.parameters()).device}")  # 确认模型参数实际位置
        return model

    elif file == './model/model_rnn':
        model = MyRnnModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        model.load_state_dict(torch.load(file, map_location=current_device, weights_only=True))
        model.eval()
        model = model.to(current_device)
        print(f"模型 {file} 已移至 {next(model.parameters()).device}")  # 确认模型参数实际位置
        return model

    # 如果文件路径不匹配，可以返回 None 或抛出错误
    return None


# 输入单字，生成的诗句格式比较规定
def generate_poem(model, input_char, max_length=31):
    # --- 从模型自身获取最可靠的设备信息 ---
    try:
        model_device = next(model.parameters()).device
        print(f"生成函数检测到模型位于: {model_device}")  # 调试信息
    except StopIteration:
        print("错误：无法从模型获取设备信息，模型可能为空或无参数。")
        return "模型加载错误"
    except AttributeError:
        print(f"错误：传入的 'model' 不是一个有效的 PyTorch 模型: {model}")
        return "模型类型错误"

    (w1, word_2_index, index_2_word) = pickle.load(open(vec_params_file, 'rb'))

    result = []
    # 检查输入字符是否在词汇表中
    try:
        initial_index = word_2_index[input_char]
        if initial_index == -1:  # 假设 -1 表示未找到 (根据你之前的代码)
            return "抱歉，词汇库中没有这个字。"
    except KeyError:
        # 如果字典里直接就没有这个key
        return "抱歉，词汇库中没有这个字。"

    # --- 直接在模型所在的设备上创建初始输入张量 ---
    x_input = torch.tensor([[initial_index]], dtype=torch.long, device=model_device)

    # 初始隐藏状态设为 None，让 LSTM/RNN 层在第一次调用时自行初始化（它们会在模型设备上创建）
    h_0 = None
    c_0 = None

    # 先把有效的起始字加入结果
    result.append(input_char)

    # --- 使用 torch.no_grad() 进行推理，可以节省显存并加速 ---
    with torch.no_grad():
        if isinstance(model, MyRnnModel):
            for i in range(max_length):
                # 确认输入和隐藏状态都在 model_device 上 (x_input 已保证, h_0 为 None 或来自上次输出)
                pre, h_0 = model(x_input, h_0)  # 模型输出 pre, h_0 也会在 model_device 上

                # --- 在 GPU 上进行采样 ---
                # pre 的形状通常是 [1, vocab_size]
                # 我们需要对 vocab_size 这个维度计算概率并采样
                probs = f.softmax(pre[0], dim=0)  # 对第一个（也是唯一一个）batch的输出计算softmax
                # 使用 multinomial 进行带权随机采样，直接在 GPU 上完成
                next_token_tensor = torch.multinomial(probs, num_samples=1)  # 输出形状为 tensor([index])，仍在 model_device

                # --- 处理采样结果 ---
                # 从 GPU 张量获取 Python 整数索引 (需要 .item() 到 CPU)
                current_index_val = next_token_tensor.item()
                pre_word = index_2_word[current_index_val]
                result.append(pre_word)

                # --- 准备下一次迭代的输入 ---
                # 直接使用 GPU 上的采样结果 tensor，只需调整形状
                # .view(1, 1) 或 .unsqueeze(0).unsqueeze(0) 都可以
                x_input = next_token_tensor.view(1, 1)  # next_token_tensor 已经在 model_device 上了

        else:  # 默认是 LSTM
            for i in range(max_length):
                pre, (h_0, c_0) = model(x_input, h_0, c_0)  # 输入输出都在 model_device

                # --- 在 GPU 上采样 ---
                probs = f.softmax(pre[0], dim=0)
                next_token_tensor = torch.multinomial(probs, num_samples=1)  # 在 model_device

                # --- 处理采样结果 ---
                current_index_val = next_token_tensor.item()  # 转到 CPU 获取索引
                pre_word = index_2_word[current_index_val]
                result.append(pre_word)

                # --- 准备下一次迭代的输入 ---
                x_input = next_token_tensor.view(1, 1)  # 复用 GPU 上的张量

    # 检查是否生成了内容（至少要多于初始字符）
    if len(result) <= 1:
        return "生成失败或长度不足，请再试一次。"  # 提供更具体的反馈

    return ''.join(result)


# mode = 3 -> test
def test(start_word, model_type=1):
    if model_type == 2:
        # 加载模型并设置为eval
        model = load_model('./model/model_rnn')
        generated_poem = generate_poem(model, start_word)
        return generated_poem
    # 默认 使用LSTM
    model = load_model('./model/model_lstm')
    generated_poem = generate_poem(model, start_word)
    return generated_poem


def is_chinese(words):
    return '\u4e00' <= words <= '\u9fff'


# def generate_random_poem(model, max_length=31, num_poems=5):
#     (w1, word_2_index, index_2_word) = pickle.load(open('vec_params.pkl', 'rb'))
#
#     poems = []
#     for _ in range(num_poems):
#         starting_word = random.choice(list(word_2_index.keys()))
#         poem = generate_poem(model, starting_word, max_length)
#         poems.append(poem)
#
#     return poems

# BLEU类用不了
# def score(type=1):
#     # 默认
#     model = load_model('./model/model_lstm')
#
#     if type == 2:
#         model = load_model('./model/model_rnn')
#
#     # 生成x首随机诗句
#     poems = generate_random_poem(model)
#     # 读取参考诗句
#     with open('./txt_dataset/song.txt', 'r', encoding="utf-8") as f:
#         reference_sentences = f.read().splitlines()
#
#     # print(reference_sentences)
#
#     reference_sentences = [s.split() for s in reference_sentences[:5]]
#     reference_sentences = [reference_sentences]
#
#     candidate_sentences = [s.split() for s in poems]
#
#     # print(poems)
#
#     print(reference_sentences)
#
#     print(candidate_sentences)
#
#     # smoothie = SmoothingFunction()
#     bleu = sacrebleu.BLEU()
#     bleu_score = bleu.corpus_score(candidate_sentences, reference_sentences).score
#     bleu_score = corpus_bleu(reference_sentences, candidate_sentences, smoothing_function=smoothie.method1)
#     print(f"BLEU-4 Score: {bleu_score: .4f}")


if __name__ == '__main__':
    # train(1)
    # train(2)

    # 模型训练
    # train(mode)

    # print(all_data)
    # all_data: numpy数组 (poem_num,poem_words_num )
    # (诗歌的数量，每首诗歌的字数) -> (_,32)
    # print(np.shape(all_data))

    # word_2_index:dict  eg: '字':1
    # print(word_2_index)
    # index_2_word: 转成跟word_2_index相似的字典
    # print(index_2_word)
    # ________________________________
    converter = OpenCC('t2s')
    if mode == 3:
        while True:
            print("请输入一个字:", end='')
            word = input().strip()
            if not word:
                print("输入为空")
                break
            word = converter.convert(word[0])
            if not is_chinese(word):
                print("请输入中文")
                continue
            else:
                # test 默认使用LSTM，设为2则使用RNN
                out_poem = test(word)
                print(out_poem)

    # ________________________________
    # score()
