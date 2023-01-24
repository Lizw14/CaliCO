# CaliCO
This is pytorch implementation for ICCV21 paper: [Calibrating Concepts and Operations: Towards Symbolic Reasoning on Real Images](https://arxiv.org/pdf/2110.00519.pdf)

## Prerequisites
The codebase is tested with Python3.8 and Pytorch1.7.

Part of this codebase is built on [NSCL](https://github.com/vacancy/NSCL-PyTorch-Release). Great thanks to the authors! Please refer to the prerequisites of NSCL codebase. Specially, install [Jacinle](https://github.com/vacancy/Jacinle):
```
git clone https://github.com/vacancy/Jacinle --recursive
export PATH=<path_to_jacinle>/bin:$PATH
```



## Data preparation
Download the [GQA dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html) (ver1.2) into `data/orig_data`.
Download the extracted images features from [this link](http://cs.jhu.edu/~zhuowan/CaliCO/sgg_features.h5) into `data/features`.
The `data` directory should look like:
```
data/orig_data/questions1.2
 - train_balanced_questions.json
 - val_balanced_questions.json
 - testdev_balanced_questions.json
data/orig_data/sceneGraphs
 - train_sceneGraphs.json
 - val_sceneGraphs.json
data/features
 - sgg_features.h5
 - sgg_info.json
...
```

## Testing
Download the trained model from [this link](http://cs.jhu.edu/~zhuowan/CaliCO/cco_trained.zip) and put it under `ckpt/cco_trained`. Then run the following command. It should reach accuracy 55.81 (on testdev split).
```
sh scripts/cco_test.sh
```

## Training
Run the following command to train:
```
sh scripts/cco_train.sh
```

## Analysis
The dataset splits (filtered by operation weights, easy/hard) as in Table 4 and Fig 5 can be downloaded [here](http://cs.jhu.edu/~zhuowan/CaliCO/analysis.zip).

## Citation
```
@InProceedings{li2021calibrating,
    author = {Li, Zhuowan and Stengel-Eskin, Elias and Zhang, Yixiao and Xie, Cihang and Tran, Quan and Van Durme, Benjamin and Yuille, Alan},
    title = {Calibrating Concepts and Operations: Towards Symbolic Reasoning on Real Images},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    year = {2021}
}
```
