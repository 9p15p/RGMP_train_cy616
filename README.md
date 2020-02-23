# RGMP PyTorch

This is forked from the official demo [code](https://github.com/seoungwugoh/RGMP) for the paper. [PDF](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1029.pdf)

Added training script with TensorBoard support.
___
## Test Environment
- Ubuntu 
- python 3.6
- Pytorch 0.3.1
  + installed with CUDA.



## How to Run Inference
1) Download [DAVIS-2017](https://davischallenge.org/davis2017/code.html).
2) Edit path for `DAVIS_ROOT` in run.py.
``` python
DAVIS_ROOT = '<Your DAVIS path>'
```
3) Download [weights.pth](https://www.dropbox.com/s/gt0kivrb2hlavi2/weights.pth?dl=0) or from [baidudownload,pwd:3fnk](https://pan.baidu.com/s/1bY-0HiQGfV3AljqL5QAT4g)and place it the same folde as run.py.
4) To run single-object video object segmentation on DAVIS-2016 validation.
``` 
python run.py
```
5) To run multi-object video object segmentation on DAVIS-2017 validation.
``` 
python run.py -MO
```
6) Results will be saved in `./results/SO` or `./results/MO`.

## How to train a model
``` python3 train.py```

## TensorBoard Support
Install [TensorBoardX](https://github.com/lanpa/tensorboard-pytorch) to view loss, IoU and generated masks in real-time during training.


  










