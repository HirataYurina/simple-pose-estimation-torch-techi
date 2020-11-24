# Simple Baselines for Human Pose Estimation and Tracking
<img src="/imgs/man5_result.jpg" width="200px">

**This is a simple module that predicts single pose estimation.**

**This prediction file only has 200 lines code. So, it is very concise.**

#### Start Predicting

**Just run predict.py.**

```shell
usage: predict.py [-h] [--model_path MODEL_PATH] [--device DEVICE]
                  [--img_path IMG_PATH] [--resize RESIZE] [--std STD]
                  [--mean MEAN] [--valid_thres VALID_THRES]
                  [--only_up_limb ONLY_UP_LIMB]

optional arguments:
  -h, --help                   show this help message and exit
  --model_path MODEL_PATH
                               the path of your model
  --device DEVICE              use cpu or gpu to infer
  --img_path IMG_PATH          path of your image
  --resize RESIZE              the input size of your model and the format is (width,
                               height)
  --std STD                    the std used to normalize your images
  --mean MEAN                  the mean used to normalize your images
  --valid_thres VALID_THRES
  --only_up_limb ONLY_UP_LIMB
```

***

:fire: pretained model url:https://pan.baidu.com/s/1Ia0UPdFY7OWowZzHdW4h5Q ​
:fire: password：0swj