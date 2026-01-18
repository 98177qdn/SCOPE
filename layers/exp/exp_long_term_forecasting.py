from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, metric_and_visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import os
from tqdm import tqdm
from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, epsilon=1e-6):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        pred, target: shape = [B, N, D]
        """
        abs_target = torch.abs(target)  # [B, N, D]

        # ✅ 每个样本的每个通道计算最大值：结果 [B, 1, D]
        max_vals = torch.amax(abs_target, dim=1, keepdim=True) + self.epsilon

        # 每个样本、每个通道独立归一化计算权重
        weights = 1.0 + self.alpha * (abs_target / max_vals)  # [B, N, D]

        # 加权 MSE
        loss = weights * (pred - target) ** 2  # [B, N, D]

        return loss.mean()


class TopKWeightedMSELoss(nn.Module):
    def __init__(self, k=6, weight_factor=5.0, channel_boost=True, eps=1e-6):
        super(TopKWeightedMSELoss, self).__init__()
        self.k = k
        self.weight_factor = weight_factor
        self.channel_boost = channel_boost  # 是否启用通道损失感知
        self.eps = eps

    def forward(self, pred, target):
        """
        pred, target: shape = [B, N, D]
        """
        B, N, D = target.shape
        abs_target = torch.abs(target)  # [B, N, D]

        # 1️⃣ Top-k 绝对值加权权重
        flat = abs_target.permute(0, 2, 1).reshape(-1, N)  # [B*D, N]
        topk_indices = torch.topk(flat, self.k, dim=1).indices  # [B*D, k]
        weights = torch.ones_like(flat)  # 初始为1
        # row_idx = torch.arange(flat.shape[0], device=pred.device).unsqueeze(1).expand(-1, self.k)
        # weights[row_idx, topk_indices] = self.weight_factor  # 设置 top-k 加权

        weights = weights.reshape(B, D, N).permute(0, 2, 1)  # [B, N, D]

        # 2️⃣ 计算每个通道的 MSE（不加权），作为通道损失感知系数
        if self.channel_boost:
            mse_per_channel = ((pred - target) ** 2).mean(dim=(0, 1))  # [D]
            # 标准化处理，让权重均值为 1，避免爆炸
            normed_weights = (mse_per_channel + self.eps) / (mse_per_channel.mean() + self.eps)  # [D]
            channel_weights = normed_weights.view(1, 1, D)  # broadcast 到 [B, N, D]
            weights = weights * channel_weights  # 每通道额外加权

        # 3️⃣ 加权 MSE 损失
        loss = weights * (pred - target) ** 2  # [B, N, D]
        return loss.mean()


class TruncatedElementwiseRelativeErrorWeightedMSELoss(nn.Module):
    def __init__(self, args,clip_value=2e-3, threshold=0.12, boost_factor=2.0, eps=1e-6):
        super().__init__()
        self.clip_value = clip_value  # 小值截断
        self.threshold = threshold  # 相对误差阈值
        self.boost_factor = args.boost_factor  # 放大系数
        self.eps = eps  # 防止除零

    def forward(self, pred, target):
        """
        pred, target: [B, N, D]
        """
        B, N, D = target.shape

        # 1️⃣ 对目标值进行截断，避免除以太小的数
        clipped_target = torch.clamp(torch.abs(target), min=self.clip_value)  # [B, N, D]

        # 2️⃣ 计算逐元素相对误差
        rel_error_map = torch.abs(pred - target) / (clipped_target + self.eps)  # [B, N, D]

        # 3️⃣ 计算每个通道的平均相对误差
        mean_rel_error_per_channel = rel_error_map.mean(dim=(0, 1))  # [D]

        # 4️⃣ 初始化通道权重为 1
        channel_weights = torch.ones_like(mean_rel_error_per_channel)  # [D]

        # 5️⃣ 对高误差通道乘以 boost_factor
        channel_weights[mean_rel_error_per_channel > self.threshold] *= self.boost_factor  # [D]

        # 6️⃣ 广播到 [B, N, D]
        channel_weights = channel_weights.view(1, 1, D)

        # 7️⃣ 加权 MSE
        loss = channel_weights * (pred - target) ** 2  # [B, N, D]

        return loss.mean()


criterion_dict = {
    'max': WeightedMSELoss,
    'mse': TopKWeightedMSELoss,
    'percent': TruncatedElementwiseRelativeErrorWeightedMSELoss,
}

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        if self.args.weighted_mse:
            print('选择截断百分比误差4')
            criterion = TruncatedElementwiseRelativeErrorWeightedMSELoss(args=self.args)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader),
                                                                          total=len(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.output_dim

                if self.args.use_behind_409_compute_loss:
                    compute_loss_len = 409
                else:
                    compute_loss_len = self.args.pred_len
                outputs = outputs[:, -compute_loss_len:, f_dim:]
                batch_y = batch_y[:, -compute_loss_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)

        best_model_path = path + '/' + 'checkpoint.pth'

        # 判断文件是否存在
        if os.path.exists(best_model_path) and self.args.load_model:
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"✅ 成功加载模型权重：{best_model_path}")
        else:
            print(f"⚠️ 模型文件不存在：{best_model_path}")

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader),
                                                                          total=len(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -self.args.output_dim

                        if self.args.use_behind_409_compute_loss:
                            compute_loss_len = 409
                        else:
                            compute_loss_len = self.args.pred_len
                        outputs = outputs[:, -compute_loss_len:, f_dim:]
                        batch_y = batch_y[:, -compute_loss_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -self.args.output_dim

                    if self.args.use_behind_409_compute_loss:
                        compute_loss_len = 409
                    else:
                        compute_loss_len = self.args.pred_len
                    outputs = outputs[:, -compute_loss_len:, f_dim:]
                    batch_y = batch_y[:, -compute_loss_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 1000 == 0 and self.args.shuchu:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test0(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + self.args.test_picture_save_path + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 缓存与chunk路径初始化
        pred_buffer, true_buffer = [], []
        chunk_index = 0
        pred_chunk_paths, true_chunk_paths = [], []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader),
                                                                          total=len(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.output_dim

                compute_loss_len = 409 if self.args.use_behind_409_compute_loss else self.args.pred_len
                outputs = outputs[:, -compute_loss_len:, :]
                batch_y = batch_y[:, -compute_loss_len:, :].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(batch_y.shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                pred_buffer.append(pred)
                true_buffer.append(true)

                if (i + 1) % 1000 == 0 or (i + 1) == len(test_loader):
                    pred_chunk = np.concatenate(pred_buffer, axis=0)
                    true_chunk = np.concatenate(true_buffer, axis=0)

                    pred_path = os.path.join(folder_path, f"pred_chunk_{chunk_index}.npy")
                    true_path = os.path.join(folder_path, f"true_chunk_{chunk_index}.npy")
                    np.save(pred_path, pred_chunk)
                    np.save(true_path, true_chunk)
                    pred_chunk_paths.append(pred_path)
                    true_chunk_paths.append(true_path)
                    pred_buffer.clear()
                    true_buffer.clear()
                    chunk_index += 1

                if i % 1000 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        input = test_data.inverse_transform(input.reshape(-1, input.shape[-1])).reshape(input.shape)
                    gt = np.concatenate((input[0, -1:, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, -1:, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 合并所有chunk
        preds = np.concatenate([np.load(p) for p in pred_chunk_paths], axis=0)
        trues = np.concatenate([np.load(t) for t in true_chunk_paths], axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 删除所有chunk文件
        for path in pred_chunk_paths + true_chunk_paths:
            os.remove(path)

        result_path = './results/' + setting + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.mean(dtw_list)
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f.write('\n\n')

        np.save(os.path.join(result_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_path, 'pred.npy'), preds)
        np.save(os.path.join(result_path, 'true.npy'), trues)

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        preds = []
        trues = []
        folder_path = './test_results/' + self.args.test_picture_save_path + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader),
                                                                          total=len(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.output_dim
                if self.args.use_behind_409_compute_loss:
                    compute_loss_len = 409
                else:
                    compute_loss_len = self.args.pred_len
                outputs = outputs[:, -compute_loss_len:, :]
                batch_y = batch_y[:, -compute_loss_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(
                        outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)).reshape(outputs.shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(batch_y.shape[0] * batch_y.shape[1], -1)).reshape(batch_y.shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 1000 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, -1:, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, -1:, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mse, percent = metric_and_visual(preds, trues, setting)
        print('mse:{}, percent:{}, dtw:{}'.format(mse, percent, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, percent:{}, dtw:{}'.format(mse, percent, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        return
