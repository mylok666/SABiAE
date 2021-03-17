import numpy as np
from collections import OrderedDict  # 获取有序的字典
import os
import glob
import skimage
import skimage.io as io
import skimage.transform as trans
import torch.utils.data as data

# Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
rng = np.random.RandomState(2020)  # 伪随机数生成器  为了确保结果一样
def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = io.imread(filename)
    image_resized = trans.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


"""DataLoder重写data.Dataset这个类"""
"""torch.utils.data.Dataset是代表自定义数据集方法的抽象类，你可以自己定义你的数据类继承这个抽象类，非常简单，只需要定义__len__和__getitem__这两个方法就可以"""
'''这里继承一个Dataset类，写一个将数据处理成DataLoader的类'''
'''需要重写len方法：提供dataset的大小；还需要重写getitem类，该方法支持从0到len(self)的索引'''


class DataLoader(data.Dataset):  # 这个功能到底是什么 生成什么？ 把视频帧变成-1到1的numpy文件
    """"下载数据 初始化数据都在这里完成"""

    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4,
                 num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()  # 类要实现的函数 得在init先声明
        self.samples = self.get_all_samples()  # 类要实现的函数 得在init先声明  而__len__ 和__getitem__必须重载

    def setup(self):  # 对视频文件准备好字典用于查询
        videos = glob.glob(os.path.join(self.dir, '*'))  # 返回所有匹配的文件路径列表
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):
                '''用ped2test作例子：ped2test里有2010帧，这里videos[01/02/03...]['frames']是对应的帧的数目，然后每个文件夹的帧都减去_time_step就是4，所以一共减去4*12帧，2010-48=1962帧'''
                frames.append(self.videos[video_name]['frame'][i])

        return frames

    def __getitem__(self, index):  # 定义获取容器中指定元素的行为，相当于self[key]，即允许类对象可以有索引操作。
        video_name = self.samples[index].split('/')[-2]  # 几号视频
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])  # 几号帧

        batch = []
        for i in range(self._time_step + self._num_pred):  # for i in range(5):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width)  # 把元数据进行调整 改成-1到1区间的npy文件
            if self.transform is not None:  # 判断变量是否是None 比如false None 空字符串 空字典 空元组都是false
                batch.append(self.transform(
                    image))  # (256,256,3) append5次 所以有15 输入的totensor后的，会把图片变成C*H*W的tensor,  所以是1962 * 15 * 256 *256

        return np.concatenate(batch, axis=0)

    def __len__(self):  # 定义当被len()函数调用时的行为（返回容器中元素的个数）
        return len(self.samples)
# dataset这个用于制作数据集