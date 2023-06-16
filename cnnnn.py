# 导入一些必要的库
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 从你给的链接中读取数据
data = pd.read_csv("data/cases.csv", encoding="utf-8")

# 把罪状转换成数字表示，用空格分隔每个字
data["罪状"] = data["罪状"].apply(lambda x: " ".join(list(x)))

# 把罪名转换成多标签的二进制表示，用0和1表示是否属于某个罪名
labels = data["罪名"].unique()
label_dict = {label: i for i, label in enumerate(labels)}
data["罪名"] = data["罪名"].apply(lambda x: [1 if label in x else 0 for label in labels])

# 划分训练集和测试集，比例为8:2
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# 定义一些超参数
vocab_size = 5000  # 词汇表的大小，即最多保留多少个不同的字
embed_size = 128  # 词嵌入的维度，即每个字用多少维的向量表示
hidden_size = 256  # LSTM的隐藏层的维度，即LSTM的输出用多少维的向量表示
num_layers = 2  # LSTM的层数，即有多少个LSTM堆叠在一起
num_classes = len(labels)  # 类别的数量，即有多少个不同的罪名
batch_size = 64  # 批次的大小，即每次训练用多少个样本
num_epochs = 10  # 训练的轮数，即对所有样本训练多少次
learning_rate = 0.01  # 学习率，即模型更新的速度


# 定义一个函数，把罪状转换成数字序列，用于输入模型
def text_to_sequence(text, vocab_size):
    # 建立一个字典，把每个字映射到一个数字，数字越小表示字出现的频率越高
    word_counts = {}
    for word in text.split():
        word_counts[word] = word_counts.get(word, 0) + 1
    word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    word_dict = {word: i + 1 for i, (word, count) in enumerate(word_counts[:vocab_size - 1])}
    # 把罪状中的每个字转换成对应的数字，如果字不在字典中，则用0表示
    sequence = [word_dict.get(word, 0) for word in text.split()]
    return sequence


# 定义一个函数，把数据集转换成批次，用于训练和测试模型
def get_batches(data, batch_size):
    # 把数据集打乱顺序
    data = data.sample(frac=1).reset_index(drop=True)
    # 计算有多少个批次
    num_batches = (len(data) - 1) // batch_size + 1
    # 对每个批次进行处理
    batches = []
    for i in range(num_batches):
        # 取出这个批次的数据
        batch_data = data[i * batch_size:(i + 1) * batch_size]
        # 把罪状和罪名转换成列表形式，并进行填充和截断，使得长度一致
        texts = batch_data["罪状"].tolist()
        labels = batch_data["罪名"].tolist()
        max_length = max(len(text.split()) for text in texts)
        texts = [text_to_sequence(text, vocab_size) for text in texts]
        texts = [text + [0] * (max_length - len(text)) for text in texts]
        texts = [text[:max_length] for text in texts]
        # 把列表转换成张量，用于输入模型
        texts = torch.tensor(texts, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)
        # 保存这个批次的数据
        batches.append((texts, labels))
    return batches


# 定义一个LSTM模型，用于对罪状进行分类
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        # 定义一个词嵌入层，把数字序列转换成词向量序列
        self.embed = nn.Embedding(vocab_size, embed_size)
        # 定义一个LSTM层，把词向量序列转换成隐藏状态序列
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # 定义一个全连接层，把最后一个隐藏状态转换成输出向量
        self.linear = nn.Linear(hidden_size, num_classes)
        # 定义一个sigmoid函数，把输出向量转换成概率值
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x的形状是(batch_size, max_length)，表示一个批次的数字序列
        # 通过词嵌入层，得到词向量序列
        x = self.embed(x)  # (batch_size, max_length, embed_size)
        # 通过LSTM层，得到隐藏状态序列和最后一个隐藏状态
        out, (h_n, c_n) = self.lstm(
            x)  # out: (batch_size, max_length, hidden_size), h_n: (num_layers, batch_size, hidden_size)
        # 取出最后一个隐藏状态，作为句子的表示
        out = h_n[-1, :, :]  # (batch_size, hidden_size)
        # 通过全连接层，得到输出向量
        out = self.linear(out)  # (batch_size, num_classes)
        # 通过sigmoid函数，得到概率值
        out = self.sigmoid(out)  # (batch_size, num_classes)
        return out


# 实例化一个LSTM模型，并定义损失函数和优化器
model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers, num_classes)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 对训练集进行训练
for epoch in range(num_epochs):
    # 获取训练集的批次数据
    train_batches = get_batches(train_data, batch_size)
    # 对每个批次进行训练
    for i, (texts, labels) in enumerate(train_batches):
        # 前向传播，得到模型的输出
        outputs = model(texts)
        # 计算损失值
        loss = criterion(outputs, labels)
        # 反向传播，更新模型的参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印训练信息
        if (i + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss {loss.item():.4f}")
    torch.save(model.state_dict(), 'model_name.pth')

# 对测试集进行测试
# 获取测试集的批次数据
test_batches = get_batches(test_data, batch_size)
# 初始化一些指标，用于评估模型的效果
test_loss = 0.0
test_acc = 0.0
test_recall = 0.0
test_f1 = 0.0
# 对每个批次进行测试

for texts, labels in test_batches:
    # 前向传播，得到模型的输出
    outputs = model(texts)
    # 计算损失值，并累加到总损失中
    loss = criterion(outputs, labels)
    test_loss += loss.item()
    # 把输出概率值转换成二进制标签，并和真实标签进行比较，计算准确率、召回率和F1值，并累加到总指标中
    outputs = outputs > 0.5
    outputs = outputs.type(torch.int).numpy()
    labels