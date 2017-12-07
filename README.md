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

TODO:
* Evaluation is a bit strange? (use the official evaluator)

| 12k steps   | EM (%) | F1 (%) | loss |
| ----------- |:------:|:------:|:----:|
| train       | 46.2   | 75.2   | 3.09 |
| dev         | 44.6   | 76.5   | 3.18 |

