import copy
import os
#import cv2
import time
import random
import numpy as np
import tensorflow as tf

from config import *
from misc.utils import *
from torchvision import datasets,transforms

class DataGenerator:

    def __init__(self, args):
        """ Data Generator

        Generates batch-iid and batch-non-iid under both
        labels-at-client and labels-at-server scenarios.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """

        self.args = args
        # 设置输出的文件夹
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task)
        self.shape = (32,32,3)

    # 生成数据
    def generate_data(self):
        print('generating {} ...'.format(self.args.task))
        start_time = time.time()
        self.task_cnt = -1
        # 根据任务 - 标签位置 和 数据是否同分布
        self.is_labels_at_server = True if 'server' in self.args.scenario else False
        self.is_imbalanced = True if 'imb' in self.args.task else False
        # 加载数据集 - 训练测试都在一起
        x, y = self.load_dataset(self.args.dataset_id)
        # 按任务产生数据集 保存为.npy到本地
        self.generate_task(x, y)
        print(f'{self.args.task} done ({time.time()-start_time}s)')

    def load_dataset(self, dataset_id):
        temp = {}
        if self.args.dataset_id_to_name[dataset_id] == 'cifar_10':
            temp['train'] = datasets.CIFAR10(self.args.dataset_path, train=True, download=True) 
            temp['test'] = datasets.CIFAR10(self.args.dataset_path, train=False, download=True)

            x, y = [], []
            # 训练测试都在一起，并存入一个变量中
            for dtype in ['train', 'test']:
                for image, target in temp[dtype]:
                    x.append(np.array(image))
                    y.append(target)

        elif self.args.dataset_id_to_name[dataset_id] == 'fashion-mnist':
            transform = transforms.Compose(
                [transforms.Resize(32),
                 transforms.ToTensor()])
            print('fashion-mnist')
            temp['train'] = datasets.FashionMNIST(self.args.dataset_path, train=True, download=True, transform=transform)
            temp['test'] = datasets.FashionMNIST(self.args.dataset_path, train=False, download=True, transform=transform)
            x, y = [], []
            # 训练测试都在一起，并存入一个变量中
            for dtype in ['train', 'test']:
                for image, target in temp[dtype]:
                    image = np.array(image)
                    image_ = np.concatenate((image, image, image), axis=0)
                    image_ = np.transpose(image_, (1, 2, 0))
                    x.append(np.array(image_))
                    y.append(target)
        else:
            print('数据集设置出错！')
            exit()

        # 按配置中的种子，对应xy随机打乱
        x, y = self.shuffle(x, y)
        print('------------',np.shape(x[0]),'-----------------------')
        print(f'{self.args.dataset_id_to_name[self.args.dataset_id]} ({np.shape(x)}) loaded.')
        return x, y

    def generate_task(self, x, y):
        # 保存测试验证集 并返回训练数据
        x_train, y_train = self.split_train_test_valid(x, y)

        # x,y均为narray
        # 是否为迪利克雷分布 - self.args.only是直接分享标记
        if self.args.diri and self.args.share and 'expert' in self.args.task and 'unbalance' in self.args.task and not self.is_imbalanced:
            print('专家标记 - 服务器数据不平衡')
            self.niid_diri_shareExpert_balanceServer(x_train, y_train)
        elif self.args.diri and self.args.share and 'expert' in self.args.task and 'balance' in self.args.task and not self.is_imbalanced:
            print('专家标记 - 服务器数据平衡')
            self.niid_diri_shareExpert_balanceServer(x_train, y_train)
        elif self.args.diri and self.args.share and 'expert' not in self.args.task:
            print('diri + share + 标签在服务器，只分享无标签')  # 弃用
            self.niid_diri_share(x_train, y_train)
        elif self.args.diri and 'unbalance' in self.args.task and not self.is_imbalanced:
            print('diri分布 - 服务器数据不平衡')
            self.niid_diri_forim(x_train,y_train)
        elif self.args.diri and not self.is_imbalanced:
            print('diri分布 - 服务器数据平衡')
            self.niid_diri(x_train, y_train)
        else:
            print('正常')
            # 按类分离标记和未标记数据
            s, u = self.split_s_and_u(x_train, y_train)
            # 标签部分 - 独立同分布的
            self.split_s(s)
            self.split_u(u)

    def split_train_test_valid(self, x, y):
        self.num_examples = len(x)
        # 训练集数量 -
        self.num_train = self.num_examples - (self.args.num_test+self.args.num_valid) 
        self.num_test = self.args.num_test
        # 赋值，标签种类
        self.labels = np.unique(y)
        # train set - 测试集设置 + 保存
        x_train = x[:self.num_train]
        y_train = y[:self.num_train]
        # test set
        x_test = x[self.num_train:self.num_train+self.num_test]
        y_test = y[self.num_train:self.num_train+self.num_test]
        y_test = tf.keras.utils.to_categorical(y_test, len(self.labels))
        l_test = np.unique(y_test)
        # 保存测试集 - 名字：test_ + 数据集名
        self.save_task({
            'x': x_test,
            'y': y_test,
            'labels': l_test,
            'name': f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        # valid set - 验证集设置 + 保存
        x_valid = x[self.num_train+self.num_test:]
        y_valid = y[self.num_train+self.num_test:]
        y_valid = tf.keras.utils.to_categorical(y_valid, len(self.labels))
        l_valid = np.unique(y_valid)
        self.save_task({
            'x': x_valid,
            'y': y_valid,
            'labels': l_valid,
            'name': f'valid_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        # 返回 训练集
        return x_train, y_train

    def split_s_and_u(self, x, y):
        # 按标签是否在服务器，设置标签数据数量
        if self.is_labels_at_server:
            self.num_s = self.args.num_labels_per_class  # -- 最终的值将乘上数据类别 5*10
        else:
            self.num_s = self.args.num_labels_per_class * self.args.num_clients 

        # 不同标签分别存储
        data_by_label = {}
        # 不同标签种类
        for label in self.labels:
            #  找到y数组中所有值等于label的元素的下标
            idx = np.where(y[:]==label)[0] 
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0
        s_by_label, u_by_label = {}, {}
        # label 作为idx ，data才存储x，y
        # 每个类都分标签 和 无标签数据并存储
        for label, data in data_by_label.items():
            s_by_label[label] = {
                'x': data['x'][:self.num_s],
                'y': data['y'][:self.num_s]
            }
            u_by_label[label] = {
                'x': data['x'][self.num_s:],
                'y': data['y'][self.num_s:]
            }
            # 总计无标签数量
            self.num_u += len(u_by_label[label]['x'])

        # 返回按类分的标记和未标记数据集
        return s_by_label, u_by_label


    def split_s(self, s):
        # 若在标签在服务器 - 不分割
        if self.is_labels_at_server:
            x_labeled = []
            y_labeled = []
            for label, data in s.items():
                #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
                x_labeled = [*x_labeled, *data['x']]
                y_labeled = [*y_labeled, *data['y']]
            x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
            self.save_task({
                'x': x_labeled,
                'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
                'labels': np.unique(y_labeled)
            })
        # 标签在客户端
        else:
            for cid in range(self.args.num_clients):
                x_labeled = []
                y_labeled = []
                for label, data in s.items():
                    start = self.args.num_labels_per_class * cid
                    end = self.args.num_labels_per_class * (cid+1)
                    _x = data['x'][start:end]
                    _y = data['y'][start:end]
                    x_labeled = [*x_labeled, *_x]
                    y_labeled = [*y_labeled, *_y]
                x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
                self.save_task({
                    'x': x_labeled,
                    'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                    'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_labeled)
                })

    def split_u(self, u):
        # 非独立同分布
        if self.is_imbalanced:
            ten_types_of_class_imbalanced_dist = [
                [0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15], # type 0
                [0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03], # type 1 
                [0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03], # type 2 
                [0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03], # type 3 
                [0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02], # type 4 
                [0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03], # type 5 
                [0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03], # type 6 
                [0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03], # type 7 
                [0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15], # type 8 
                [0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50], # type 9
            ]
            labels = list(u.keys())
            num_u_per_client = int(self.num_u/self.args.num_clients)
            offset_per_label = {label:0 for label in labels}
            for cid in range(self.args.num_clients):
                # batch-imbalanced
                x_unlabeled = []
                y_unlabeled = []
                dist_type = cid%len(labels)
                freqs = np.random.choice(labels, num_u_per_client, p=ten_types_of_class_imbalanced_dist[dist_type])
                frq = []
                for label, data in u.items():
                    num_instances = len(freqs[freqs==label])
                    frq.append(num_instances)
                    start = offset_per_label[label]
                    end = offset_per_label[label]+num_instances
                    x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                    y_unlabeled = [*y_unlabeled, *data['y'][start:end]] 
                    offset_per_label[label] = end
                x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_unlabeled)
                })    
        else:
            # batch-iid
            for cid in range(self.args.num_clients):
                x_unlabeled = []
                y_unlabeled = []
                for label, data in u.items():
                    # print('>>> ', label, len(data['x']))
                    num_unlabels_per_class = int(len(data['x'])/self.args.num_clients)
                    start = num_unlabels_per_class * cid
                    end = num_unlabels_per_class * (cid+1)
                    x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                    y_unlabeled = [*y_unlabeled, *data['y'][start:end]]  
                x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_unlabeled)
                })

    def save_task(self, data):
        np_save(base_dir=self.base_dir, filename=f"{data['name']}.npy", data=data)
        # 打印数据集名 + 存储后数据的内不同标签数量 + 数据集中数据量
        print(f"filename:{data['name']}, labels:[{','.join(map(str, data['labels']))}], num_examples:{len(data['x'])}")

    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random.seed(self.args.seed)
        random.shuffle(idx)
        return np.array(x)[idx], np.array(y)[idx]


    # train_labels 标签list
    def diri(self, train_labels, alpha, n_clients):
        n_classes = train_labels.max() + 1
        label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
        # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

        class_idcs = [np.argwhere(train_labels==y).flatten()
               for y in range(n_classes)]
        # 记录每个K个类别对应的样本下标

        client_idcs = [[] for _ in range(n_clients)]
        # 记录N个client分别对应样本集合的索引
        for c, fracs in zip(class_idcs, label_distribution):
            # np.split按照比例将类别为k的样本划分为了N个子集
            # for i, idcs 为遍历第i个client对应样本集合的索引
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs


    def niid_diri(self, x, y):
        # 若在标签在服务器
        if self.is_labels_at_server:
            print('标签在服务器')
            s, u = self.split_s_and_u(x, y)
            # 分割标签部分
            self.split_s(s)
            # 分割无标签部分
            x_unlabeled = []
            y_unlabeled = []
            for label, data in u.items():
                #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]
            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
            for i in range(self.args.num_clients):
                # 无标签部分
                x_unlabeled_, y_unlabeled_ = x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]]
                self.save_task({
                    'x': x_unlabeled_,
                    'y': tf.keras.utils.to_categorical(y_unlabeled_, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_)
                })

        # 若标签在客户端
        else:
            print('标签在客户端')
            client_idcs = self.diri(y, self.args.alpha, self.args.num_clients)

            for i in range(self.args.num_clients):
                # 标签数据量
                s_fac = int(self.args.fac * len(client_idcs[i]))
                # 有标签部分
                x_temp, y_temp = self.shuffle(x[client_idcs[i]], y[client_idcs[i]])
                x_labeled, y_labeled = x_temp[:s_fac], y_temp[:s_fac]

                self.save_task({
                    'x': x_labeled,
                    'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                    'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_labeled)
                })

                # 无标签部分 - 如果要设置标记部分也进行无标记训练的话，就不需要去除x[client_idcs[i][s_fac:]], y_temp[client_idcs[i][s_fac:]]
                # client_idcs的下标缩影是基于x,y的，不能在x_temp中运用
                x_unlabeled, y_unlabeled = x_temp[s_fac:], y_temp[s_fac:]
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled)
                })

    def niid_diri_share(self, x, y):
        # 若在标签在服务器
        if self.is_labels_at_server:
            print('标签在服务器')
            s, u = self.split_s_and_u(x, y)
            # 分割标签部分
            self.split_s(s)
            # 分割无标签部分
            x_unlabeled = []
            y_unlabeled = []
            for label, data in u.items():
                #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]
            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)

            share_x, share_y = [], []
            # 存储每个用户x,y
            x_unlabeled_client, y_unlabeled_client = [], []
            for i in range(self.args.num_clients):
                x_unlabeled_client.extend([]), y_unlabeled_client.extend([])

            # 设置分享
            for i in range(self.args.num_clients):
                share_ = int(self.args.share_rate * len(client_idcs[i]))
                # 无标签部分 - 获取原始部分,并打乱 打乱后就不能再用client_idcs
                temp_x, temp_y = self.shuffle(x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]])
                x_unlabeled_client.append(copy.deepcopy(temp_x))
                y_unlabeled_client.append(copy.deepcopy(temp_y))
                # 存入分享list - 去前share_个
                share_x.extend(x_unlabeled_client[i][:share_])
                share_y.extend(y_unlabeled_client[i][:share_])
                # 去除分享部分，防止后面重复
                x_unlabeled_client[i] = x_unlabeled_client[i][share_:]
                y_unlabeled_client[i] = y_unlabeled_client[i][share_:]

            for i in range(self.args.num_clients):
                # 添加分享部分
                x_unlabeled_client[i] = np.concatenate((x_unlabeled_client[i], share_x), axis=0)
                y_unlabeled_client[i] = np.concatenate((y_unlabeled_client[i], share_y), axis=0)
                self.save_task({
                    'x': x_unlabeled_client[i],
                    'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_client[i])
                })

        # 若标签在客户端
        else:
            print('标签在客户端')
            client_idcs = self.diri(y, self.args.alpha, self.args.num_clients)

            x_labeled_client, y_labeled_client = [], []
            x_unlabeled_client, y_unlabeled_client = [], []
            # for i in range(self.args.num_clients):
            #     x_labeled_client.append([]), y_labeled_client.append([])
            #     x_unlabeled_client.append([]), y_unlabeled_client.append([])
            share_x_labeled, share_y_labeled = [], []
            share_x_unlabeled, share_y_unlabeled = [], []
            for i in range(self.args.num_clients):
                # x_share_labeled_temp, y_share_labeled_temp = [], []
                # x_share_unlabeled_temp, y_share_unlabeled_temp = [], []
                # 标签数据量
                s_fac = int(self.args.fac * len(client_idcs[i]))
                # 分享量
                share_num_labeled = int(self.args.share_rate * s_fac)
                share_num_unlabeled = int(self.args.share_rate * (len(client_idcs[i]) - s_fac))

                x_temp, y_temp = self.shuffle(x[client_idcs[i]], y[client_idcs[i]])

                # 有标签部分
                #x_labeled_client[i], y_labeled_client[i] = x_temp[:s_fac], y_temp[:s_fac]
                x_labeled_client.append(x_temp[:s_fac])
                y_labeled_client.append(y_temp[:s_fac])
                x_share_labeled_temp = x_labeled_client[i][:share_num_labeled]
                y_share_labeled_temp = y_labeled_client[i][:share_num_labeled]
                share_x_labeled.extend(x_share_labeled_temp)
                share_y_labeled.extend(y_share_labeled_temp)

                x_labeled_client[i], y_labeled_client[i] = \
                    x_labeled_client[i][share_num_labeled:], y_labeled_client[i][share_num_labeled:]

                # 无标签部分
                x_unlabeled_client.append(x_temp[s_fac:])
                y_unlabeled_client.append(y_temp[s_fac:])
                x_share_unlabeled_temp = x_unlabeled_client[i][:share_num_unlabeled]
                y_share_unlabeled_temp = y_unlabeled_client[i][:share_num_unlabeled]
                share_x_unlabeled.extend(x_share_unlabeled_temp)
                share_y_unlabeled.extend(y_share_unlabeled_temp)

                x_unlabeled_client[i], y_unlabeled_client[i] = \
                    x_unlabeled_client[i][share_num_unlabeled:], y_unlabeled_client[i][share_num_unlabeled:]

            for i in range(self.args.num_clients):
                # x_labeled_client[i].extend(share_x_labeled)
                # y_labeled_client[i].extend(share_y_labeled)
                x_labeled_client[i] = np.concatenate([x_labeled_client[i], share_x_labeled],axis=0)
                y_labeled_client[i] = np.concatenate([y_labeled_client[i], share_y_labeled], axis=0)
                self.save_task({
                    'x': x_labeled_client[i],
                    'y': tf.keras.utils.to_categorical(y_labeled_client[i], len(self.labels)),
                    'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_labeled_client[i])
                })

                # 无标签部分 - 如果要设置标记部分也进行无标记训练的话，就不需要去除x[client_idcs[i][s_fac:]], y_temp[client_idcs[i][s_fac:]]
                # client_idcs的下标缩影是基于x,y的，不能在x_temp中运用
                # x_unlabeled_client[i].extend(share_x_unlabeled)
                # y_unlabeled_client[i].extend(share_y_unlabeled)
                x_unlabeled_client[i] = np.concatenate([x_unlabeled_client[i], share_x_unlabeled],axis=0)
                y_unlabeled_client[i] = np.concatenate([y_unlabeled_client[i], share_y_unlabeled], axis=0)
                self.save_task({
                    'x': x_unlabeled_client[i],
                    'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_client[i])
                })

    # def niid_diri_shareOnly(self, x, y):
    #     # 若在标签在服务器
    #     if self.is_labels_at_server:
    #         print('标签在服务器')
    #         print('暂时没写该场景下只分享标签的情况')
    #         return
    #         s, u = self.split_s_and_u(x, y)
    #         # 分割标签部分
    #         self.split_s(s)
    #         # 分割无标签部分
    #         x_unlabeled = []
    #         y_unlabeled = []
    #         for label, data in u.items():
    #             #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
    #             x_unlabeled = [*x_unlabeled, *data['x']]
    #             y_unlabeled = [*y_unlabeled, *data['y']]
    #         x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
    #         client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
    #
    #         share_x, share_y = [], []
    #         # 存储每个用户x,y
    #         x_unlabeled_client, y_unlabeled_client = [], []
    #         for i in range(self.args.num_clients):
    #             x_unlabeled_client.extend([]), y_unlabeled_client.extend([])
    #
    #         # 设置分享
    #         for i in range(self.args.num_clients):
    #             share_ = int(self.args.share_rate * len(client_idcs[i]))
    #             # 无标签部分 - 获取原始部分,并打乱 打乱后就不能再用client_idcs
    #             temp_x, temp_y = self.shuffle(x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]])
    #             x_unlabeled_client.append(copy.deepcopy(temp_x))
    #             y_unlabeled_client.append(copy.deepcopy(temp_y))
    #             # 存入分享list - 去前share_个
    #             share_x.extend(x_unlabeled_client[i][:share_])
    #             share_y.extend(y_unlabeled_client[i][:share_])
    #             # 去除分享部分，防止后面重复
    #             x_unlabeled_client[i] = x_unlabeled_client[i][share_:]
    #             y_unlabeled_client[i] = y_unlabeled_client[i][share_:]
    #
    #         for i in range(self.args.num_clients):
    #             # 添加分享部分
    #             x_unlabeled_client[i] = np.concatenate((x_unlabeled_client[i], share_x), axis=0)
    #             y_unlabeled_client[i] = np.concatenate((y_unlabeled_client[i], share_y), axis=0)
    #             self.save_task({
    #                 'x': x_unlabeled_client[i],
    #                 'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
    #                 'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
    #                 'labels': np.unique(y_unlabeled_client[i])
    #             })
    #
    #     # 若标签在客户端
    #     else:
    #         print('标签在客户端')
    #         client_idcs = self.diri(y, self.args.alpha, self.args.num_clients)
    #
    #         x_labeled_client, y_labeled_client = [], []
    #         x_unlabeled_client, y_unlabeled_client = [], []
    #
    #         share_x_labeled, share_y_labeled = [], []
    #
    #         for i in range(self.args.num_clients):
    #             # 标签数据量
    #             s_fac = int(self.args.fac * len(client_idcs[i]))
    #             # 分享量
    #             share_num_labeled = int(self.args.share_rate * s_fac)
    #
    #             x_temp, y_temp = self.shuffle(x[client_idcs[i]], y[client_idcs[i]])
    #
    #             # 有标签部分
    #             x_labeled_client.append(x_temp[:s_fac])
    #             y_labeled_client.append(y_temp[:s_fac])
    #             # 有标记部分中抽取分享数据
    #             x_share_labeled_temp = x_labeled_client[i][:share_num_labeled]
    #             y_share_labeled_temp = y_labeled_client[i][:share_num_labeled]
    #             share_x_labeled.extend(x_share_labeled_temp)
    #             share_y_labeled.extend(y_share_labeled_temp)
    #             # 去除原有的标签
    #             x_labeled_client[i], y_labeled_client[i] = \
    #                 x_labeled_client[i][share_num_labeled:], y_labeled_client[i][share_num_labeled:]
    #
    #             # 无标签部分
    #             x_unlabeled_client.append(x_temp[s_fac:])
    #             y_unlabeled_client.append(y_temp[s_fac:])
    #             # x_share_unlabeled_temp = x_unlabeled_client[i][:share_num_unlabeled]
    #             # y_share_unlabeled_temp = y_unlabeled_client[i][:share_num_unlabeled]
    #             # share_x_unlabeled.extend(x_share_unlabeled_temp)
    #             # share_y_unlabeled.extend(y_share_unlabeled_temp)
    #
    #             # x_unlabeled_client[i], y_unlabeled_client[i] = \
    #             #     x_unlabeled_client[i][share_num_unlabeled:], y_unlabeled_client[i][share_num_unlabeled:]
    #
    #         for i in range(self.args.num_clients):
    #             # x_labeled_client[i].extend(share_x_labeled)
    #             # y_labeled_client[i].extend(share_y_labeled)
    #             # 有标签数据
    #             x_labeled_client[i] = np.concatenate([x_labeled_client[i], share_x_labeled], axis=0)
    #             y_labeled_client[i] = np.concatenate([y_labeled_client[i], share_y_labeled], axis=0)
    #
    #             self.save_task({
    #                 'x': x_labeled_client[i],
    #                 'y': tf.keras.utils.to_categorical(y_labeled_client[i], len(self.labels)),
    #                 'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
    #                 'labels': np.unique(y_labeled_client[i])
    #             })
    #
    #             # 无标签部分 - 如果要设置标记部分也进行无标记训练的话，就不需要去除x[client_idcs[i][s_fac:]], y_temp[client_idcs[i][s_fac:]]
    #             # client_idcs的下标缩影是基于x,y的，不能在x_temp中运用
    #             # x_unlabeled_client[i].extend(share_x_unlabeled)
    #             # y_unlabeled_client[i].extend(share_y_unlabeled)
    #             # x_unlabeled_client[i] = np.concatenate([x_unlabeled_client[i], share_x_unlabeled],axis=0)
    #             # y_unlabeled_client[i] = np.concatenate([y_unlabeled_client[i], share_y_unlabeled], axis=0)
    #             # 直接赋值 不分享
    #             self.save_task({
    #                 'x': x_unlabeled_client[i],
    #                 'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
    #                 'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
    #                 'labels': np.unique(y_unlabeled_client[i])
    #             })

    def niid_diri_shareExpert_balanceServer(self, x, y):
        # 若在标签在服务器
        if self.is_labels_at_server:
            print('标签在服务器')
            s, u = self.split_s_and_u(x, y)
            # 服务器标签设置
            x_labeled, y_labeled = [], [] # 作为服务器用于存放标签数据
            for label, data in s.items():
                #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
                x_labeled = [*x_labeled, *data['x']]
                y_labeled = [*y_labeled, *data['y']]

            # 无标签数据设置
            x_unlabeled, y_unlabeled = [], []
            for label, data in u.items():
                #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]

            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
            x_unlabeled_client, y_unlabeled_client = [], []  # 客户端的无标签数据

            share_x_labeled, share_y_labeled = [], []  # 分享的无标签数据转化为便签数据

            for i in range(self.args.num_clients):
                # 分享量
                share_num_labeled = int(self.args.share_rate * len(client_idcs[i]))
                x_temp, y_temp = self.shuffle(x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]])

                # 每个客户端数据
                x_unlabeled_client.append(x_temp)
                y_unlabeled_client.append(y_temp)

                # 抽取分享数据 - 无标签中抽取
                x_share_labeled_temp = x_unlabeled_client[i][:share_num_labeled]
                y_share_labeled_temp = y_unlabeled_client[i][:share_num_labeled]
                share_x_labeled.extend(x_share_labeled_temp)
                share_y_labeled.extend(y_share_labeled_temp)

            final_x_labeled = np.concatenate([x_labeled, share_x_labeled], axis=0)
            final_y_labeled = np.concatenate([y_labeled, share_y_labeled], axis=0)

            # 标签数据设置
            self.save_task({
                'x': final_x_labeled,
                'y': tf.keras.utils.to_categorical(final_y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
                'labels': np.unique(final_y_labeled)
            })

            # 无标签数据设置
            for i in range(self.args.num_clients):
                self.save_task({
                    'x': x_unlabeled_client[i],
                    'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_client[i])
                })


    def split_s_and_u_forim(self, x, y):
        # 按标签是否在服务器，设置标签数据数量
        if self.is_labels_at_server:
            self.num_s = self.args.num_labels_per_class  # -- 最终的值将乘上数据类别 5*10
            print("服务器标签数据量：", self.num_s)
        else:
            self.num_s = self.args.num_labels_per_class * self.args.num_clients
        print('服务端数据不平衡')
        num_rate = [0.05, 0.05, 0.05, 0.05,
                    0.05, 0.05, 0.15, 0.15,
                    0.2, 0.2]
        num_label = []
        for i in range(10):
            num_label.append(int(self.num_s * 10 * num_rate[i]))
        # 不同标签分别存储
        data_by_label = {}
        # 不同标签种类
        for label in self.labels:
            #  找到y数组中所有值等于label的元素的下标
            idx = np.where(y[:]==label)[0]
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0
        s_by_label, u_by_label = {}, {}
        # label 作为idx ，data才存储x，y
        # 每个类都分标签 和 无标签数据并存储
        for label, data in data_by_label.items():
            s_by_label[label] = {
                'x': data['x'][:num_label[label]],
                'y': data['y'][:num_label[label]]
            }
            u_by_label[label] = {
                'x': data['x'][num_label[label]:],
                'y': data['y'][num_label[label]:]
            }
            # 总计无标签数量
            self.num_u += len(u_by_label[label]['x'])

        # 返回按类分的标记和未标记数据集
        return s_by_label, u_by_label

    def niid_diri_forim(self, x, y):
        # 若在标签在服务器
        if self.is_labels_at_server:
            print('标签在服务器')
            s, u = self.split_s_and_u_forim(x, y)
            # 分割标签部分
            self.split_s(s)
            # 分割无标签部分
            x_unlabeled = []
            y_unlabeled = []
            for label, data in u.items():
                #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]
            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
            for i in range(self.args.num_clients):
                # 无标签部分
                x_unlabeled_, y_unlabeled_ = x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]]
                self.save_task({
                    'x': x_unlabeled_,
                    'y': tf.keras.utils.to_categorical(y_unlabeled_, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_)
                })

    def niid_diri_shareExpert_imbalanceServer(self, x, y):
        # 若在标签在服务器
        if self.is_labels_at_server:
            print('标签在服务器')
            s, u = self.split_s_and_u_forim(x, y)
            # 服务器标签设置
            x_labeled, y_labeled = [], [] # 作为服务器用于存放标签数据
            for label, data in s.items():
                #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
                x_labeled = [*x_labeled, *data['x']]
                y_labeled = [*y_labeled, *data['y']]

            # 无标签数据设置
            x_unlabeled, y_unlabeled = [], []
            for label, data in u.items():
                #  *用于解包data字典中的x和y列表，然后将它们的元素添加到x_labeled和y_labeled列表中
                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]

            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
            x_unlabeled_client, y_unlabeled_client = [], []  # 客户端的无标签数据

            share_x_labeled, share_y_labeled = [], []  # 分享的无标签数据转化为便签数据

            for i in range(self.args.num_clients):
                # 分享量
                share_num_labeled = int(self.args.share_rate * len(client_idcs[i]))
                x_temp, y_temp = self.shuffle(x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]])

                # 每个客户端数据
                x_unlabeled_client.append(x_temp)
                y_unlabeled_client.append(y_temp)

                # 抽取分享数据 - 无标签中抽取
                x_share_labeled_temp = x_unlabeled_client[i][:share_num_labeled]
                y_share_labeled_temp = y_unlabeled_client[i][:share_num_labeled]
                share_x_labeled.extend(x_share_labeled_temp)
                share_y_labeled.extend(y_share_labeled_temp)

            final_x_labeled = np.concatenate([x_labeled, share_x_labeled], axis=0)
            final_y_labeled = np.concatenate([y_labeled, share_y_labeled], axis=0)

            # 标签数据设置
            self.save_task({
                'x': final_x_labeled,
                'y': tf.keras.utils.to_categorical(final_y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
                'labels': np.unique(final_y_labeled)
            })

            # 无标签数据设置
            for i in range(self.args.num_clients):
                self.save_task({
                    'x': x_unlabeled_client[i],
                    'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_client[i])
                })

    # def niid_diri_shareExpert_imbalanceServer(self, x, y):
    #     # 若在标签在服务器
    #     if self.is_labels_at_server:
    #         print('标签在服务器')
    #         client_idcs = self.diri(y, self.args.alpha, self.args.num_clients + 1)
    #
    #         x_labeled_client, y_labeled_client = [], []  # 作为服务器用于存放标签数据
    #         x_unlabeled_client, y_unlabeled_client = [], []  # 客户端的无标签数据
    #
    #         share_x_labeled, share_y_labeled = [], []  # 分享的无标签数据转化为便签数据
    #
    #         for i in range(self.args.num_clients):
    #             # 分享量
    #             share_num_labeled = int(self.args.share_rate * len(client_idcs[i]))
    #             x_temp, y_temp = self.shuffle(x[client_idcs[i]], y[client_idcs[i]])
    #
    #             # 每个客户端数据
    #             x_unlabeled_client.append(x_temp)
    #             y_unlabeled_client.append(y_temp)
    #
    #             # 抽取分享数据 - 无标签中抽取
    #             x_share_labeled_temp = x_unlabeled_client[i][:share_num_labeled]
    #             y_share_labeled_temp = y_unlabeled_client[i][:share_num_labeled]
    #             share_x_labeled.extend(x_share_labeled_temp)
    #             share_y_labeled.extend(y_share_labeled_temp)
    #
    #         final_x_labeled = np.concatenate([x_labeled_client[self.args.num_clients], share_x_labeled], axis=0)
    #         final_y_labeled = np.concatenate([y_labeled_client[self.args.num_clients], share_y_labeled], axis=0)
    #
    #         # 标签数据设置
    #         self.save_task({
    #             'x': final_x_labeled,
    #             'y': tf.keras.utils.to_categorical(final_y_labeled, len(self.labels)),
    #             'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
    #             'labels': np.unique(final_y_labeled)
    #         })
    #
    #         # 无标签数据设置
    #         for i in range(self.args.num_clients):
    #             self.save_task({
    #                 'x': x_unlabeled_client[i],
    #                 'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
    #                 'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
    #                 'labels': np.unique(y_unlabeled_client[i])
    #             })
