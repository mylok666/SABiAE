# SABiAE
Pytorch implementation of my second paper：Self Attention based Bi-directional Long Short-Term Memory Auto Encoder.

This implementation used Nonlocal module.

I only trained the ped2 and avenue datasets,the results:
method    |  ped2   |  avenue
:-----  |-------: |:----:
SABiAE    | 95.6    |  84.2

**Prepare**
-----------
Download the ped2 and the avenue datasets.
[BaiduYun : njnu](https://pan.baidu.com/s/1kq6NNFFeqxY9esx-YmR58g )

Modify dataset_path in Train.py, and then unzip the datasets under your data root.

My trained model is in './exp/ped2/ped2e-4/model_196.000_0.00001_0.00002.pth'

Train
-----

python Train.py --dataset_type 'ped2' --dataset_path 'your_path' --epochs 200 --batch_size 1 --exp_dir './exp/'

Eva
----

python Eva.py --dataset_type 'ped2' --dataset_path 'yourpath' --exp_dir 'your exp dir' --model_dir './exp/ped2/ped2e-4/model_196.000_0.00001_0.00002.pth'
