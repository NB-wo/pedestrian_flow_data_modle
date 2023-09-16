import datetime
import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from gen_data import delete_anomaly_from_database, get_data_from_database, get_latest_data_from_database


def encode_cyclic_feature(data, max_val):
    """
    对周期性特征进行正弦和余弦编码。
    """
    sin_encoded = np.sin(2 * np.pi * data / max_val)
    cos_encoded = np.cos(2 * np.pi * data / max_val)
    return sin_encoded, cos_encoded

def preprocess_data(data):
    """
    对数据进行预处理。
    """
    
    # 对年份进行简单的归一化 (考虑到年份不是周期性的)
    scaled_year = data[:, 0] / 3000.0  # 假设3000年是一个大概的上限
    
    # 对月、日、小时、分钟、秒进行正弦和余弦编码
    scaled_month_sin, scaled_month_cos = encode_cyclic_feature(data[:, 1], 12)
    scaled_day_sin, scaled_day_cos = encode_cyclic_feature(data[:, 2], 31)
    scaled_hour_sin, scaled_hour_cos = encode_cyclic_feature(data[:, 3], 24)
    scaled_minute_sin, scaled_minute_cos = encode_cyclic_feature(data[:, 4], 60)
    scaled_second_sin, scaled_second_cos = encode_cyclic_feature(data[:, 5], 60)
    
    # 对流量数据进行归一化
    scaler = RobustMinMaxScaler(feature_range=(0, 1))
    scaled_flow_counts = scaler.fit_transform(data[:, 6].reshape(-1, 1))
    
    # 合并所有经过处理的特征
    scaled_data = np.vstack((
        scaled_year,
        scaled_month_sin, scaled_month_cos,
        scaled_day_sin, scaled_day_cos,
        scaled_hour_sin, scaled_hour_cos,
        scaled_minute_sin, scaled_minute_cos,
        scaled_second_sin, scaled_second_cos,
        scaled_flow_counts.ravel()
    )).T
    
    # 为LSTM模型准备数据
    look_back = 60  # 设定回溯窗口大小
    x_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i])
        y_train.append(scaled_data[i, -1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # 注意，现在x_train的shape是[num_samples, look_back, 13]，因为我们有13个特征
    
    # 转换为 PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # 返回处理后的数据
    return x_train_tensor, y_train_tensor

class RobustMinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, data):
        self.min_ = np.min(data)
        self.max_ = np.max(data)

    def transform(self, data):
        # Ensure data is within the range of min_ and max_
        data = np.clip(data, self.min_, self.max_)
        
        # Normalize data
        return (data - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class LSTM(nn.Module):
    """
    LSTM模型定义。
    
    参数:
    - input_size (int): 输入序列的特征数量。默认值为12。
    - hidden_layer_size (int): LSTM隐藏层的大小。默认值为100。
    - output_size (int): 输出的大小。默认为1。
    """
    
    def __init__(self, input_size=12, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        # 全连接层，用于输出预测
        self.linear = nn.Linear(hidden_layer_size, output_size)

        # LSTM的初始隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        """
        前向传播
        
        参数:
        - input_seq (torch.Tensor): 输入的序列数据。
        
        返回:
        - torch.Tensor: 对序列最后一个时间步的预测。
        """
        # 通过LSTM层
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        
        # 将LSTM的输出传入全连接层进行预测
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        
        # 返回序列的最后一个预测
        return predictions[-1]

class EnhancedLSTM(nn.Module):
    # 更好的模型
    def __init__(self, input_size=12, projection_size=64, hidden_layer_size=128, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # 线性投影层
        self.projection = nn.Linear(input_size, projection_size)

        # LSTM层
        self.lstm = nn.LSTM(projection_size, hidden_layer_size)

        # 全连接层，用于输出预测
        self.linear = nn.Linear(hidden_layer_size, output_size)

        # LSTM的初始隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # 通过线性投影层
        projected_input = self.projection(input_seq)
        
        # 通过LSTM层
        lstm_out, self.hidden_cell = self.lstm(projected_input.view(len(projected_input), 1, -1), self.hidden_cell)
        
        # 将LSTM的输出传入全连接层进行预测
        predictions = self.linear(lstm_out.view(len(projected_input), -1))
        
        # 返回序列的最后一个预测
        return predictions[-1]


def train_model(data, model, optimizer, loss_function, epochs=300):
    """
    训练LSTM模型的函数。
    
    参数:
    - data (array-like): 输入的数据。
    - model (nn.Module): 使用的LSTM模型。
    - optimizer (torch.optim.Optimizer): 优化器。
    - loss_function (nn.Module): 损失函数。
    - epochs (int): 训练的轮数。默认值为10。
    """
    # 数据预处理
    x_train_tensor, y_train_tensor = preprocess_data(data)
    
    # 开始训练
    for i in tqdm(range(epochs)):
        for seq, labels in zip(x_train_tensor, y_train_tensor):
            
            # 重置梯度
            optimizer.zero_grad()
            
            # 重置LSTM的隐藏状态和单元状态
            #model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
            #                     torch.zeros(1, 1, model.hidden_layer_size))
            
            # 获取模型的预测结果
            y_pred = model(seq).view(-1)
            
            # 计算损失
            single_loss = loss_function(y_pred, labels)
            
            # 反向传播
            single_loss.backward()
            
            # 更新权重
            optimizer.step()

def predict_future(model, test_inputs, future_predict=20):
    """
    使用LSTM模型预测未来的值。
    
    参数:
    - model (nn.Module): 已经训练过的LSTM模型。
    - test_inputs (list): 输入的测试数据。
    - future_predict (int): 需要预测的未来值的数量。默认值为20。
    
    返回:
    - list: 预测的未来值。
    """
    model.eval()  # 将模型设置为评估模式（关闭dropout等）
    
    for i in range(future_predict):
        # 获取最后60个数据点作为输入
        seq = torch.FloatTensor(test_inputs[-60:])
        
        # 预测下一个值
        with torch.no_grad():
            # 重置隐藏状态和单元状态
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            
            # 使用模型进行预测
            next_value = model(seq).item()
            
            # 将预测值添加到输入序列中，以供下一次预测使用
            test_inputs.append(next_value)
    
    # 返回预测的未来值
    return test_inputs[-future_predict:]

# 定义一个阈值
ANOMALY_THRESHOLD = 0.1  # 这个值可以根据你的实际需求进行调整

def detect_anomaly(predicted, actual):
    """
    检测是否存在异常
    """
    difference = abs(predicted - actual)
    if difference > ANOMALY_THRESHOLD:
        trigger_alert(predicted, actual, difference)

def trigger_alert(predicted, actual, difference):
    """
    触发异常警告
    """
    print(f"Anomaly Detected! Predicted: {predicted}, Actual: {actual}, Difference: {difference}")

if __name__ == "__main__":
    # 初始设置
    model = LSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()

    # 从数据库加载初始数据（处理时排除ID）
    data = get_data_from_database(host="localhost", user="mysql", password="123456", database="flow_of_people")[:, 1:]
    train_model(data, model, optimizer, loss_function, epochs=10)
    
    # 获取数据集中的最后一个日期（排除ID）
    last_training_date = data[-1][0]
    
    while True:
        # 从数据库获取最新数据
        latest_data = get_latest_data_from_database(host="localhost", user="mysql", password="123456", database="flow_of_people")
        
        # 检查是否有新数据
        if latest_data is not None:
            # 存储ID，以备后续可能的异常删除
            latest_data_id = latest_data[0]
            # 提取日期
            latest_data_date = datetime.date(latest_data[1], latest_data[2], latest_data[3])
            # 处理数据，排除ID
            latest_data_processed = np.array([latest_data[1:]])
            
            # 使用模型进行预测
            prediction = predict_future(model, preprocess_data(latest_data_processed), future_predict=1)
            
            # 检查异常
            anomaly = detect_anomaly(prediction, latest_data[-1])  # 假设最后一列是实际的计数
            if anomaly:
                # 从数据库中删除异常值
                delete_anomaly_from_database(host="localhost", user="mysql", password="123456", database="flow_of_people", anomaly_id=latest_data_id)
            
            # 检查是否需要重新训练（每3天一次）
            if (latest_data_date - last_training_date).days >= 3:
                # 从数据库中重新获取数据，排除ID
                data = get_data_from_database(host="localhost", user="mysql", password="123456", database="flow_of_people")[:, 1:]
                train_model(data, model, optimizer, loss_function, epochs=10)
                last_training_date = latest_data_date
            else:
                time.sleep(300)