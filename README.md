[MobileNetV2](https://arxiv.org/abs/1801.04381) implementation using PyTorch

#### Steps

* Configure `imagenet` path by changing `data_dir` in `main.py`
* `python main.py --benchmark` for model information
* `bash ./main.sh $ --train` for training model, `$` is number of GPUs
* `python main.py --test` for testing

#### Note

* `bash ./main.sh 2 --train --batch-size 256` for reproducing results
* MobileNetV2-Large achieved 72.7 % top-1 and 90.8 % top-5 after 450 epochs

```
Number of parameters: 3487816
Time per operator type:
        25.0655 ms.    90.7755%. Conv
        2.10124 ms.    7.60971%. Clip
       0.322608 ms.    1.16834%. FC
       0.106036 ms.   0.384014%. Add
      0.0153448 ms.  0.0555717%. AveragePool
     0.00188663 ms. 0.00683249%. Flatten
        27.6126 ms in Total
FLOP per operator type:
       0.598989 GFLOP.    99.5385%. Conv
       0.002561 GFLOP.   0.425581%. FC
    0.000216384 GFLOP.  0.0359582%. Add
       0.601766 GFLOP in Total
Feature Memory Read per operator type:
        35.8909 MB.    83.9532%. Conv
        5.12912 MB.    11.9976%. FC
        1.73107 MB.    4.04918%. Add
        42.7511 MB in Total
Feature Memory Written per operator type:
        26.7124 MB.    96.8475%. Conv
       0.865536 MB.    3.13805%. Add
          0.004 MB.  0.0145022%. FC
         27.582 MB in Total
Parameter Memory per operator type:
        8.75904 MB.    63.0917%. Conv
          5.124 MB.    36.9083%. FC
              0 MB.          0%. Add
         13.883 MB in Total
```

