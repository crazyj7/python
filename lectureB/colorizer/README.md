흑백 영상을 컬러로 변환한다.

color.py ; 이미지 변환.

video.py ; 동영상 변환. 시간이 오래 걸림.




# Colorizer

Colorize black and white (grayscale) image (or video) to colorful color with OpenCV (DNN module)
  
- Used pretrained model from [Zhang's Github](https://github.com/richzhang/colorization)  
- Original paper: [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)


![result.jpg](https://github.com/kairess/colorizer/raw/master/img/result.jpg)  
  

Project contains:  
1. Colorize image (Jupyter notebook)  
2. Colorize video (Python script)  
  

# Requirement
- Python
- OpenCV 3.4.2+
- Numpy
  

# Usage
### Colorize Image

Look at [colorize.ipynb](https://github.com/kairess/colorizer/blob/master/colorize.ipynb)
  

### Colorize Video

```
sh get_models.sh
python video.py
```
  