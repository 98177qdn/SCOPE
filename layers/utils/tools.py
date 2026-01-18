import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib import rcParams
from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(100, 20), dpi=300)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def metric_and_visual(pred,true,model_id,visiual=True):
    print("开始评估和可视化：")
    # ✅ 设置中文字体和负号正常显示
    # 设置字体为 Linux 下常见字体
    rcParams['font.family'] = 'DejaVu Sans'
    # 避免中文乱码（如使用了中文标签）
    rcParams['axes.unicode_minus'] = False

    column_list = ['2-8', '3-8', '4-8', '5-8', '2-1', '3-1', '5-1', '2-2', '3-2', '4-2', '5-2', '2-3', '4-3', '5-3',
                   '2-4',
                   '3-4', '4-4', '5-4', '3-5', '4-5', '5-5', '2-6', '3-6', '4-6', '5-6', '2-7', '3-7', '4-7', '5-7',
                   '3-3',
                   '4-1', '2-5', '左4', '左中', '左2', '右2', '右中', '右4', '电机1', '电机2', '电机3', '电机4', '左1',
                   '左3', '左5', '右1', '右3', '右5', '电机5', '电机6', '电机7', '水箱下1', '水箱下2', '水箱下3',
                   '水箱下4',
                   '水箱下5', '水箱下6', '水箱下7', '水箱上1', '水箱上2', '水箱上3', '电机8']

    pic_save_folder = os.path.join('./result_pic/', model_id)

    os.makedirs(pic_save_folder, exist_ok=True)


    # ✅ 自定义要绘制的样本索引和通道索引
    sample_indices = [1000, 2000, 3000, 4000, 10000]  # 你想画的样本，比如第 0、2、4 个样本
    channel_indices = range(pred.shape[2])  # 你想画的通道，比如通道 0 和通道 1

    if visiual:
        # ✅ 绘图
        for channel_idx in tqdm(channel_indices, desc='正在遍历传感器画图'):
            for sample_idx in sample_indices:
                sensor_name = column_list[channel_idx]
                sensor_save_folder = os.path.join(pic_save_folder, sensor_name)
                os.makedirs(sensor_save_folder, exist_ok=True)
                pic_save_path = os.path.join(sensor_save_folder, f'sample_{sample_idx}.png')

                plt.figure(figsize=(30, 10))
                plt.plot(true[sample_idx, :, channel_idx], label='true', color='blue')
                plt.plot(pred[sample_idx, :, channel_idx], label='pred', color='red', linestyle='--')
                plt.title(f"channel {sample_idx} pred and true(channel {channel_idx})")
                plt.xlabel("time")
                plt.ylabel("value")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                plt.savefig(pic_save_path)
                plt.close()
    column_list = column_list[:pred.shape[2]]
    # ✅ 设置平滑常数（防止除以0或接近0）
    epsilon = 2e-3  # 可根据实际调整

    # ✅ 条件分母：当 |true| < epsilon 时，加上 epsilon
    denominator = np.where(np.abs(true) < epsilon, np.abs(true) + epsilon, np.abs(true))

    # ✅ 百分比误差（只对小于阈值的加 epsilon）
    percentage_error = np.abs(pred - true) / denominator * 100
    print("百分比误差数组的形状：", percentage_error.shape)

    percentage_per_sample = np.mean(percentage_error, axis=1)  # 每个样本
    percentage_per_channel = np.mean(percentage_per_sample, axis=0)  # 每个通道
    percentage_avg = np.mean(percentage_per_channel)

    print("每个通道的 百分比误差（|true| < ε 时才加 epsilon）：")
    print(percentage_per_channel)
    print(f"✅ 所有通道平均百分比误差（条件加 {epsilon}）：", f"{percentage_avg:.2e}")

    # ✅ MSE（不排除任何点）
    squared_error = (pred - true) ** 2
    mse_per_sample = np.mean(squared_error, axis=1)
    mse_per_channel = np.mean(mse_per_sample, axis=0)
    mse_avg = np.mean(mse_per_channel)

    print("每个通道的 MSE：")
    print(mse_per_channel)
    print("✅ 所有通道平均 MSE：", f"{mse_avg:.2e}")


    # ✅ 创建 DataFrame（每个通道）
    result_df = pd.DataFrame({
        "壳体传感器": column_list,
        "百分比误差": percentage_per_channel,
        "MSE（不加截断）": mse_per_channel,
    })

    save_dir = os.path.join(f'./results_mse_db_summary/',model_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'percent_mse_summary.csv')
    result_df.to_csv(save_path)

    print(f"\n✅ 结果已保存至：{save_path}")

    return mse_avg,percentage_avg