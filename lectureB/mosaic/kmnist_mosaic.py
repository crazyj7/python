'''

kmnist download

https://www.kaggle.com/anokas/kuzushiji


샘플 이미지의 픽셀 평균값과  원본 픽셀값이 비슷한 것으로 채운다.
하얀 이미지는 하얀 픽셀에...
검은 이미지는 검은 픽셀에...

평균값 히스토그램 분포를 확인. 90~255
120~245 부분을 사용한다..
원본 이미지의 픽셀 값들의 분포를 저 범위로 normalize한다.
cv2.normalize... alpha=min , beta=max. (120~245)

평균값이 120~245까지 해당되는 패치들을 분리함. 랜덤하게 나중에 선택. 없으면 다음 빈스에서 검색.


'''
