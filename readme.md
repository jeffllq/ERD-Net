=====

This repository contains the code for [ERD-Net: Modeling Entity and Relation Dynamics for Temporal Knowledge Graph Reasoning]
## Data

4 datasets were used in the paper:

- YAGO, WIKI, ICEWS14, GDELT: could be found in (https://github.com/INK-USC/RE-Net/tree/master/data)

## Requirements
  * PyTorch 2.0
  * Python 3.9


### Installation

#### process data, get statistics for repeatted patterns in history
```sh
python get_history_record.py --d ?
```
as for negative sampling process, look out in data_util.py



#### Intrinsic embeddings learning:
```sh
python main.py --dataset ? --max-epochs 20 --mode pretrain --lr 0.001 --device-id ?
```

#### time dynamic embeddings learning:
```sh
python main.py --dataset ? --max-epochs 40 --mode train --lr 0.001 --use-bias 0 --freq-info 0 --device-id ? --train-history-len ?
python main.py --dataset ? --max-epochs 40 --mode train --lr 0.001 --use-bias 1 --freq-info 0 --device-id ? --train-history-len ?
```

#### analyze generation mode：
```sh
python main.py --dataset ? --max-epochs 40 --mode train --lr 0.001 --use-bias 0 --freq-info 3 --alpha 0.? --device-id ? --train-history-len ?
python main.py --dataset ? --max-epochs 40 --mode train --lr 0.001 --use-bias 1 --freq-info 3 --alpha 0.? --device-id ? --train-history-len ?
```

```
alpha：
   YAGO 0.9
   WIKI 0.9
ICEWS14 0.4
  GDELT 0.6
```



