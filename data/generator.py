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

        self.args = args
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task)
        self.shape = (32,32,3)


    def generate_data(self):
        print('generating {} ...'.format(self.args.task))
        start_time = time.time()
        self.task_cnt = -1

        self.is_labels_at_server = True if 'server' in self.args.scenario else False
        self.is_imbalanced = True if 'imb' in self.args.task else False

        x, y = self.load_dataset(self.args.dataset_id)

        self.generate_task(x, y)
        print(f'{self.args.task} done ({time.time()-start_time}s)')

    def load_dataset(self, dataset_id):
        temp = {}
        if self.args.dataset_id_to_name[dataset_id] == 'cifar_10':
            temp['train'] = datasets.CIFAR10(self.args.dataset_path, train=True, download=True) 
            temp['test'] = datasets.CIFAR10(self.args.dataset_path, train=False, download=True)

            x, y = [], []
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


        x, y = self.shuffle(x, y)

        print(f'{self.args.dataset_id_to_name[self.args.dataset_id]} ({np.shape(x)}) loaded.')
        return x, y

    def generate_task(self, x, y):

        x_train, y_train = self.split_train_test_valid(x, y)

        if self.args.diri and self.args.share and 'expert' in self.args.task and 'unbalance' in self.args.task and not self.is_imbalanced:
            self.niid_diri_shareExpert_balanceServer(x_train, y_train)
        elif self.args.diri and self.args.share and 'expert' in self.args.task and 'balance' in self.args.task and not self.is_imbalanced:
            self.niid_diri_shareExpert_balanceServer(x_train, y_train)
        elif self.args.diri and self.args.share and 'expert' not in self.args.task:
            self.niid_diri_share(x_train, y_train)
        elif self.args.diri and 'unbalance' in self.args.task and not self.is_imbalanced:
            self.niid_diri_forim(x_train,y_train)
        elif self.args.diri and not self.is_imbalanced:
            self.niid_diri(x_train, y_train)
        else:
            s, u = self.split_s_and_u(x_train, y_train)
            self.split_s(s)
            self.split_u(u)

    def split_train_test_valid(self, x, y):
        self.num_examples = len(x)

        self.num_train = self.num_examples - (self.args.num_test+self.args.num_valid) 
        self.num_test = self.args.num_test

        self.labels = np.unique(y)

        x_train = x[:self.num_train]
        y_train = y[:self.num_train]
        # test set
        x_test = x[self.num_train:self.num_train+self.num_test]
        y_test = y[self.num_train:self.num_train+self.num_test]
        y_test = tf.keras.utils.to_categorical(y_test, len(self.labels))
        l_test = np.unique(y_test)

        self.save_task({
            'x': x_test,
            'y': y_test,
            'labels': l_test,
            'name': f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })

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

        return x_train, y_train

    def split_s_and_u(self, x, y):

        if self.is_labels_at_server:
            self.num_s = self.args.num_labels_per_class
        else:
            self.num_s = self.args.num_labels_per_class * self.args.num_clients 


        data_by_label = {}

        for label in self.labels:

            idx = np.where(y[:]==label)[0] 
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0
        s_by_label, u_by_label = {}, {}

        for label, data in data_by_label.items():
            s_by_label[label] = {
                'x': data['x'][:self.num_s],
                'y': data['y'][:self.num_s]
            }
            u_by_label[label] = {
                'x': data['x'][self.num_s:],
                'y': data['y'][self.num_s:]
            }

            self.num_u += len(u_by_label[label]['x'])


        return s_by_label, u_by_label


    def split_s(self, s):

        if self.is_labels_at_server:
            x_labeled = []
            y_labeled = []
            for label, data in s.items():

                x_labeled = [*x_labeled, *data['x']]
                y_labeled = [*y_labeled, *data['y']]
            x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
            self.save_task({
                'x': x_labeled,
                'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
                'labels': np.unique(y_labeled)
            })

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

        print(f"filename:{data['name']}, labels:[{','.join(map(str, data['labels']))}], num_examples:{len(data['x'])}")

    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random.seed(self.args.seed)
        random.shuffle(idx)
        return np.array(x)[idx], np.array(y)[idx]



    def diri(self, train_labels, alpha, n_clients):
        n_classes = train_labels.max() + 1
        label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)


        class_idcs = [np.argwhere(train_labels==y).flatten()
               for y in range(n_classes)]


        client_idcs = [[] for _ in range(n_clients)]

        for c, fracs in zip(class_idcs, label_distribution):

            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs


    def niid_diri(self, x, y):

        if self.is_labels_at_server:

            s, u = self.split_s_and_u(x, y)

            self.split_s(s)

            x_unlabeled = []
            y_unlabeled = []
            for label, data in u.items():

                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]
            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
            for i in range(self.args.num_clients):

                x_unlabeled_, y_unlabeled_ = x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]]
                self.save_task({
                    'x': x_unlabeled_,
                    'y': tf.keras.utils.to_categorical(y_unlabeled_, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_)
                })


        else:

            client_idcs = self.diri(y, self.args.alpha, self.args.num_clients)

            for i in range(self.args.num_clients):

                s_fac = int(self.args.fac * len(client_idcs[i]))

                x_temp, y_temp = self.shuffle(x[client_idcs[i]], y[client_idcs[i]])
                x_labeled, y_labeled = x_temp[:s_fac], y_temp[:s_fac]

                self.save_task({
                    'x': x_labeled,
                    'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                    'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_labeled)
                })


                x_unlabeled, y_unlabeled = x_temp[s_fac:], y_temp[s_fac:]
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled)
                })

    def niid_diri_share(self, x, y):

        if self.is_labels_at_server:

            s, u = self.split_s_and_u(x, y)

            self.split_s(s)

            x_unlabeled = []
            y_unlabeled = []
            for label, data in u.items():

                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]
            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)

            share_x, share_y = [], []

            x_unlabeled_client, y_unlabeled_client = [], []
            for i in range(self.args.num_clients):
                x_unlabeled_client.extend([]), y_unlabeled_client.extend([])


            for i in range(self.args.num_clients):
                share_ = int(self.args.share_rate * len(client_idcs[i]))

                temp_x, temp_y = self.shuffle(x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]])
                x_unlabeled_client.append(copy.deepcopy(temp_x))
                y_unlabeled_client.append(copy.deepcopy(temp_y))

                share_x.extend(x_unlabeled_client[i][:share_])
                share_y.extend(y_unlabeled_client[i][:share_])

                x_unlabeled_client[i] = x_unlabeled_client[i][share_:]
                y_unlabeled_client[i] = y_unlabeled_client[i][share_:]

            for i in range(self.args.num_clients):

                x_unlabeled_client[i] = np.concatenate((x_unlabeled_client[i], share_x), axis=0)
                y_unlabeled_client[i] = np.concatenate((y_unlabeled_client[i], share_y), axis=0)
                self.save_task({
                    'x': x_unlabeled_client[i],
                    'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_client[i])
                })


        else:

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

                s_fac = int(self.args.fac * len(client_idcs[i]))

                share_num_labeled = int(self.args.share_rate * s_fac)
                share_num_unlabeled = int(self.args.share_rate * (len(client_idcs[i]) - s_fac))

                x_temp, y_temp = self.shuffle(x[client_idcs[i]], y[client_idcs[i]])


                #x_labeled_client[i], y_labeled_client[i] = x_temp[:s_fac], y_temp[:s_fac]
                x_labeled_client.append(x_temp[:s_fac])
                y_labeled_client.append(y_temp[:s_fac])
                x_share_labeled_temp = x_labeled_client[i][:share_num_labeled]
                y_share_labeled_temp = y_labeled_client[i][:share_num_labeled]
                share_x_labeled.extend(x_share_labeled_temp)
                share_y_labeled.extend(y_share_labeled_temp)

                x_labeled_client[i], y_labeled_client[i] = \
                    x_labeled_client[i][share_num_labeled:], y_labeled_client[i][share_num_labeled:]


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



    def niid_diri_shareExpert_balanceServer(self, x, y):

        if self.is_labels_at_server:

            s, u = self.split_s_and_u(x, y)

            x_labeled, y_labeled = [], []
            for label, data in s.items():

                x_labeled = [*x_labeled, *data['x']]
                y_labeled = [*y_labeled, *data['y']]


            x_unlabeled, y_unlabeled = [], []
            for label, data in u.items():

                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]

            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
            x_unlabeled_client, y_unlabeled_client = [], []

            share_x_labeled, share_y_labeled = [], []

            for i in range(self.args.num_clients):

                share_num_labeled = int(self.args.share_rate * len(client_idcs[i]))
                x_temp, y_temp = self.shuffle(x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]])


                x_unlabeled_client.append(x_temp)
                y_unlabeled_client.append(y_temp)


                x_share_labeled_temp = x_unlabeled_client[i][:share_num_labeled]
                y_share_labeled_temp = y_unlabeled_client[i][:share_num_labeled]
                share_x_labeled.extend(x_share_labeled_temp)
                share_y_labeled.extend(y_share_labeled_temp)

            final_x_labeled = np.concatenate([x_labeled, share_x_labeled], axis=0)
            final_y_labeled = np.concatenate([y_labeled, share_y_labeled], axis=0)


            self.save_task({
                'x': final_x_labeled,
                'y': tf.keras.utils.to_categorical(final_y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
                'labels': np.unique(final_y_labeled)
            })


            for i in range(self.args.num_clients):
                self.save_task({
                    'x': x_unlabeled_client[i],
                    'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_client[i])
                })


    def split_s_and_u_forim(self, x, y):

        if self.is_labels_at_server:
            self.num_s = self.args.num_labels_per_class

        else:
            self.num_s = self.args.num_labels_per_class * self.args.num_clients

        num_rate = [0.05, 0.05, 0.05, 0.05,
                    0.05, 0.05, 0.15, 0.15,
                    0.2, 0.2]
        num_label = []
        for i in range(10):
            num_label.append(int(self.num_s * 10 * num_rate[i]))

        data_by_label = {}

        for label in self.labels:

            idx = np.where(y[:]==label)[0]
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0
        s_by_label, u_by_label = {}, {}

        for label, data in data_by_label.items():
            s_by_label[label] = {
                'x': data['x'][:num_label[label]],
                'y': data['y'][:num_label[label]]
            }
            u_by_label[label] = {
                'x': data['x'][num_label[label]:],
                'y': data['y'][num_label[label]:]
            }

            self.num_u += len(u_by_label[label]['x'])


        return s_by_label, u_by_label

    def niid_diri_forim(self, x, y):

        if self.is_labels_at_server:

            s, u = self.split_s_and_u_forim(x, y)

            self.split_s(s)

            x_unlabeled = []
            y_unlabeled = []
            for label, data in u.items():

                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]
            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
            for i in range(self.args.num_clients):

                x_unlabeled_, y_unlabeled_ = x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]]
                self.save_task({
                    'x': x_unlabeled_,
                    'y': tf.keras.utils.to_categorical(y_unlabeled_, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_)
                })

    def niid_diri_shareExpert_imbalanceServer(self, x, y):

        if self.is_labels_at_server:

            s, u = self.split_s_and_u_forim(x, y)

            x_labeled, y_labeled = [], []
            for label, data in s.items():

                x_labeled = [*x_labeled, *data['x']]
                y_labeled = [*y_labeled, *data['y']]


            x_unlabeled, y_unlabeled = [], []
            for label, data in u.items():

                x_unlabeled = [*x_unlabeled, *data['x']]
                y_unlabeled = [*y_unlabeled, *data['y']]

            x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
            client_idcs = self.diri(y_unlabeled, self.args.alpha, self.args.num_clients)
            x_unlabeled_client, y_unlabeled_client = [], []

            share_x_labeled, share_y_labeled = [], []

            for i in range(self.args.num_clients):

                share_num_labeled = int(self.args.share_rate * len(client_idcs[i]))
                x_temp, y_temp = self.shuffle(x_unlabeled[client_idcs[i]], y_unlabeled[client_idcs[i]])


                x_unlabeled_client.append(x_temp)
                y_unlabeled_client.append(y_temp)


                x_share_labeled_temp = x_unlabeled_client[i][:share_num_labeled]
                y_share_labeled_temp = y_unlabeled_client[i][:share_num_labeled]
                share_x_labeled.extend(x_share_labeled_temp)
                share_y_labeled.extend(y_share_labeled_temp)

            final_x_labeled = np.concatenate([x_labeled, share_x_labeled], axis=0)
            final_y_labeled = np.concatenate([y_labeled, share_y_labeled], axis=0)


            self.save_task({
                'x': final_x_labeled,
                'y': tf.keras.utils.to_categorical(final_y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
                'labels': np.unique(final_y_labeled)
            })


            for i in range(self.args.num_clients):
                self.save_task({
                    'x': x_unlabeled_client[i],
                    'y': tf.keras.utils.to_categorical(y_unlabeled_client[i], len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{i}',
                    'labels': np.unique(y_unlabeled_client[i])
                })


