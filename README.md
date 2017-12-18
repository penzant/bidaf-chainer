# bidaf-chainer
A chainer implementation of [Bi-directional Attention Flow for Reading Comprehension (arxiv:1611.01603)](https://arxiv.org/abs/1611.01603)

Requirements:
* Python 3.5+ (verified)
* Chainer 3.1.0+ (verified)
* The preprocessed SQuAD data from [the original BiDAF (github)](https://github.com/allenai/bi-att-flow)

To run,
```
python train.py
```

For debugging with a small data:
```
python train.py --debug_mode
```

##### Result (not official evaluation)

| dataset | EM (%) | F1 (%) | loss |
| ----------- |:------:|:------:|:----:|
| train (this repo, 10k) | 65.3   | 79.3   | 1.83 |
| dev (this repo, 10k) | 55.1   | 70.1   | 2.81 |
| dev (original) | (67.7) | (77.3) | (-) |

