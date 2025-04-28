import torch.nn as nn
import torch
import numpy as np
import logging
import os
import errno
from libs.Args import args
import matplotlib.pyplot as plt


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def reverse(tensor):
    idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    return tensor[idx]


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2
    res = None
    if len(data.size()) == 3:
        res = data[:, :, :last_dim], data[:, :, last_dim:]

    if len(data.size()) == 2:
        res = data[:, :last_dim], data[:, last_dim:]
    return res


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert (start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                             torch.linspace(start[i], end[i], n_points)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def create_net(n_inputs, n_outputs, n_layers=1,
               n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def get_logger(logpath):
    # 起手式，获得一个logger对象
    logger = logging.getLogger()
    # 该日志的级别为INFO，意味着INFO及以上级别的日志会被记录
    logger.setLevel(logging.INFO)

    # 检查之前的logger,如果没有才会重建
    if not logger.handlers:
        # 文件处理器
        # 保存log到文件夹
        file_handler = logging.FileHandler(logpath)
        # 设置文件保存级别
        file_handler.setLevel(logging.INFO)
        # 指定输出格式
        # 分别为时间，log名称，日志级别，日志
        """
        logger.debug("这是调试信息")
        logger.info("这是普通信息")
        logger.warning("这是警告信息")
        logger.error("这是错误信息")
        logger.critical("这是严重错误信息")
        """
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # 将文件处理对象加入日志对象中
        logger.addHandler(file_handler)

        # 控制台处理器，将日志输出到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info(f"Logging to {logpath}")

    return logger


def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr


def normalize_tensor(tensor):
    # 找到 Tensor 的最小值和最大值
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # 计算归一化后的 Tensor
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor


def denormalize_tensor_for_q(normalized_tensor, tensor):
    # 找到 Tensor 的最小值和最大值
    min_val = torch.min(tensor)/4
    # print("This is denormalizing, ", tensor)
    max_val = torch.max(tensor)/4
    # 恢复归一化前的 Tensor
    original_tensor = normalized_tensor * (max_val - min_val) + min_val

    return original_tensor

def denormalize_tensor_for_m(normalized_tensor, tensor):
    # 找到 Tensor 的最小值和最大值
    min_val = torch.min(tensor)
    # print("This is denormalizing, ", tensor)
    max_val = torch.max(tensor)
    # 恢复归一化前的 Tensor
    original_tensor = normalized_tensor * (max_val - min_val) + min_val

    return original_tensor


def get_next_batch(train_data, time_tp, tp_to_predict, aggregated_labels, val_labels):
    # Make the union of all time points and perform normalization across the whole dataset
    data_train = train_data.__next__()
    ob_tp = time_tp.__next__()
    batch_dict = {"observed_data": data_train, "observed_tp": ob_tp, "data_to_predict": None,
                  "tp_to_predict": tp_to_predict, "val_labels": val_labels, "agg_labels_dataloader": aggregated_labels}

    # remove the time points where there are no observations in this batch
    print("Batch data shape is", batch_dict["observed_data"].size())
    print('Aggreation label data shape are', batch_dict['agg_labels_dataloader'].shape)
    # batch_dict[ "data_to_predict"] = None
    # batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

    # batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
    # batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]

    # print("data_to_predict")
    # print(batch_dict["data_to_predict"].size())

    # if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
    # 	batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][:, non_missing_tp]

    # batch_dict["labels"] = data_dict["labels"]

    # batch_dict["mode"] = data_dict["mode"]
    return batch_dict


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            # if the path exits,error won't occur
            if e.errno != errno.EXIST:
                raise


# CustomDataset to handle high_freq_data and low_freq_data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, high_freq_data, low_freq_data):
        self.high_freq_data = high_freq_data
        self.low_freq_data = low_freq_data

    def __len__(self):
        # Return the length of high frequency data
        return len(self.high_freq_data)

    def __getitem__(self, idx):
        # Fetch the high and low frequency data at the given index
        high_freq_sample = self.high_freq_data[idx]
        # You might need to do some processing to match high_freq_sample with low_freq_sample
        # For now, assuming low_freq_data is a subset that aligns with high_freq_data
        low_freq_sample = self.low_freq_data[idx // args.F]  # Adjust based on your freq_ratio
        return high_freq_sample, low_freq_sample

def viz_check(data):
    plt.figure()
    plt.plot(data)
    plt.title('Data check')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()


if __name__ == "__main__":
    from pre_processing import get_data
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    batch_size = 32

    data, _, label_m, label_m_no_nor = get_data(4)  # Ensure get_data() returns the correct label_m
    dataset = torch.tensor(data, dtype=torch.float32)  # No need to add batch dimension here
    label_m_tensor = torch.tensor(label_m, dtype=torch.float32).unsqueeze(
        -1)  # Ensure label_m has the correct shape [504, 1]
    # 到这里标签数据没有问题
    dataset = TensorDataset(dataset, label_m_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(dataset, data_loader)

    print(dataset[0][0])




    # # 创建一个图形
    # plt.figure()
    #
    # # 绘制数据
    # plt.subplot(1, 3, 1)  # (行数, 列数, 当前子图索引)
    # plt.plot(a, label='before_norm')
    # plt.title('Before norm')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    #
    #
    # # 添加标题和标签
    # plt.subplot(1, 3, 2)  # (行数, 列数, 当前子图索引)
    # plt.plot(a_nor, label='before_norm')
    # plt.title('After norm')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    #
    # plt.subplot(1, 3, 3)  # (行数, 列数, 当前子图索引)
    # plt.plot(a_nor_denor, label='before_norm')
    # plt.title('After denorm')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    #
    # # 显示图形
    # plt.show()

