'''

cifar10 download
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

컬러이미지 비교를 위해 색상 수를 먼저 줄이고 (quantization )
이를 위해서 K-Means Clustering 을 사용하여 32개의 컬러수로 줄인다. (군집화)
원본이미지의 셀들을 K-Means로 예측하여 결과색으로  칠한다. qunatization 완료.

이후에 패치들을 해당 픽셀에 넣는다.


'''