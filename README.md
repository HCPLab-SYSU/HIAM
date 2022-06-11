# Online Metro Origin-Destination Prediction via Heterogeneous Information Aggregation
This is a PyTorch implementation of **Online Metro Origin-Destination Prediction via Heterogeneous Information Aggregation**. 

With high trip efficiencies and cheap ticket charges, metro has recently become a popular travel mode for urban residents. Due to its significant applications, metro ridership prediction has recently attracted extensive attention in both academic and industrial communities. However, most of conventional works were merely proposed for station-level prediction, i.e., forecasting the inflow and outflow of each metro station. Such information of inflow/outflow ridership is too coarse to reflect the mobility of passengers. 

We propose a novel **Heterogeneous Information Aggregation Machine** to facilitate the online metro origin-destination prediction. To the best of our knowledge, our **HIAM** is the first deep learning based approach that fully aggregates heterogeneous information to jointly forecast the future OD ridership and DO ridership.

If you use this code for your research, please cite our papers.
```
@article{liu2022online,
  title={Online Metro Origin-Destination Prediction via Heterogeneous Information Aggregation},
  author={Liu, Lingbo and Zhu, Yuying and Li, Guanbin and Wu, Ziyi and Bai, Lei and Lin, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```

### Requirements
- python3
- numpy
- yaml
- pytorch
- torch_geometric
### Extract dataset
Please download the dataset and extract it to `data` folder.
- [Dropbox link](https://www.dropbox.com/sh/4pgk4uez7g200fg/AACHN6wMhjq_v0R2ZZ8ZeI6ma?dl=0)
- [Baidu Netdisk，password：q6e0 ](https://pan.baidu.com/s/1PHN8SNT3jTroX0sTWHsrXw)

## Train
- SHMOD
```
python train.py --config data/config/train_sh_dim76_units96_h4c512.yaml
```

- HZMOD
```
python train.py --config data/config/train_hz_dim26_units96_h4c512.yaml
```
## Test
First of all, download the trained model and extract it to the path:`data/checkpoint`.

- SHMOD
```
python evaluate.py --config data/checkpoint/eval_sh_dim76_units96_h4c512.yaml
```
- HZMOD
```
python evaluate.py --config data/checkpoint/eval_hz_dim26_units96_h4c512.yaml
```

