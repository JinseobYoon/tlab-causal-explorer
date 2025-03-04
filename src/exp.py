import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.ns_transformer import Model
from src.runner import data_provider
from src.utils import metric, visual


class ExpMain(object):
    def __init__(self, args):
        if isinstance(args, dict):
            self.args = SimpleNamespace(**args)  # Convert dict to object-like namespace
        else:
            self.args = args  # Assume it's already an object
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, args):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model_dict = {
            'NSTransformer': Model,
        }
        model = model_dict[self.args.model](self.args).float()

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
        return criterion

    # def vali(self, vali_data, vali_loader, criterion):
    #     """
    #     Validation 데이터에 대해 모델의 평균 손실(MSE)을 계산합니다.
    #     """
    #     total_loss = []
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()  # target은 CPU에서 연산 후, .to(self.device)로 이동
    #
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input 구성: label_len 만큼의 실제 값을 앞부분에 붙이고, 나머지는 0으로 채움
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #
    #             # 모델 forward (AMP 사용 여부에 따라)
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                 else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #
    #             # f_dim 결정 (features가 MS이면 마지막 차원 사용, 아니면 첫 번째 차원)
    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #
    #             loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
    #             total_loss.append(loss)
    #     avg_loss = np.average(total_loss)
    #     self.model.train()  # 다시 training 모드로 전환
    #     return avg_loss
    #
    # def train(self, setting):
    #     """
    #     오직 train 데이터에 대해 학습만 수행합니다.
    #     평가(Validation, Test)는 별도 함수로 분리합니다.
    #     """
    #     train_data, train_loader = self._get_data(flag='train')
    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #
    #     time_now = time.time()
    #     train_steps = len(train_loader)
    #     model_optim = self._select_optimizer()
    #     criterion = self._select_criterion()
    #
    #     if self.args.use_amp:
    #         scaler = torch.cuda.amp.GradScaler()
    #
    #     for epoch in range(self.args.train_epochs):
    #         iter_count = 0
    #         train_loss = []
    #         self.model.train()
    #         epoch_time = time.time()
    #
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    #             iter_count += 1
    #             model_optim.zero_grad()
    #
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input 구성
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #
    #             # 모델 forward
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #                     f_dim = -1 if self.args.features == 'MS' else 0
    #                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #                     batch_y_slice = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #                     loss = criterion(outputs, batch_y_slice)
    #                     train_loss.append(loss.item())
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                 else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #                 f_dim = -1 if self.args.features == 'MS' else 0
    #                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #                 batch_y_slice = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #                 loss = criterion(outputs, batch_y_slice)
    #                 train_loss.append(loss.item())
    #
    #             if (i + 1) % 100 == 0:
    #                 print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                 speed = (time.time() - time_now) / iter_count
    #                 left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #                 print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                 iter_count = 0
    #                 time_now = time.time()
    #
    #             if self.args.use_amp:
    #                 scaler.scale(loss).backward()
    #                 scaler.step(model_optim)
    #                 scaler.update()
    #             else:
    #                 loss.backward()
    #                 model_optim.step()
    #
    #         epoch_duration = time.time() - epoch_time
    #         avg_train_loss = np.average(train_loss)
    #         print("Epoch: {} | Train Loss: {:.7f} | Time: {:.2f}s".format(epoch + 1, avg_train_loss, epoch_duration))
    #         self.adjust_learning_rate(model_optim, epoch + 1, self.args)
    #
    #     # 학습 후 마지막 모델을 반환 (checkpoint 저장은 필요에 따라 추가)
    #     return self.model
    #
    # def test(self, setting, test=0):
    #     """
    #     Test set에 대해 모델을 평가하고, 예측 결과 및 평가 지표를 저장합니다.
    #     """
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
    #
    #     preds = []
    #     trues = []
    #     folder_path = os.path.join('./test_results', setting)
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                 else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #
    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #             batch_y_slice = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y_slice = batch_y_slice.detach().cpu().numpy()
    #
    #             preds.append(outputs)
    #             trues.append(batch_y_slice)
    #
    #             if i % 20 == 0:
    #                 input_np = batch_x.detach().cpu().numpy()
    #                 gt = np.concatenate((input_np[0, :, -1], batch_y_slice[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input_np[0, :, -1], outputs[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, f"{i}.pdf"))
    #
    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #
    #     result_folder = os.path.join('./results', setting)
    #     if not os.path.exists(result_folder):
    #         os.makedirs(result_folder)
    #
    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mape: {:.4f}, mae: {:.4f}'.format(mape, mae))
    #     with open("result.txt", 'a') as f:
    #         f.write(setting + "\n")
    #         f.write('mape: {:.4f}, mae: {:.4f}'.format(mape, mae))
    #         f.write("\n\n")
    #
    #     np.save(os.path.join(result_folder, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(os.path.join(result_folder, 'pred.npy'), preds)
    #     np.save(os.path.join(result_folder, 'true.npy'), trues)
    #
    #     return
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

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
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)  # batch 개수를 저장
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):  # 하나의 Batch씩 데이터를 불러옴
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)  # batch_x -> batch 안에 있는 각 sample의 입력값(sequence_length)
                batch_y = batch_y.float().to(self.device)  # batch_y -> batch 안에 있는 pred의 실제값(pred_length)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # batch_y에 대한 예측
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(
                    self.device)  # label_len 만큼 데이터를 앞에 붙여 사용

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0  ### output은 시점이 겹치는 경우 어떻게 처리하는가? -> 최신 예측값 사용용
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 역전파 및 가중치 업데이트
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # 조기 종료 및 학습률 조정
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

            self.adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def train(self, setting):
        # 오직 train 데이터만 사용하여 학습
        train_data, train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input 구성
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 모델 forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
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

            epoch_duration = time.time() - epoch_time
            avg_train_loss = np.average(train_loss)
            print("Epoch: {} | Train Loss: {:.7f} | Time: {:.2f}s".format(epoch + 1, avg_train_loss, epoch_duration))

            self.adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 저장된 checkpoint 대신, 마지막 모델을 저장 또는 사용
        # (필요에 따라 checkpoint 저장 로직을 추가하세요)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mape:{}, mae:{}'.format(mape, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mape:{}, mae:{}'.format(mape, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


