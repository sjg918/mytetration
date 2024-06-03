# mytetration
DMTPark 복소수 테트레이션 연산 수렴/발산 집합 시각화</br>
주요구현: 다중GPU를사용한병렬처리, 자동edge추적을통한영상딸깍생성</br>
사용된library: Pytorch</br>
1개의 영상을 처리하는데 필요한 시간</br>
RTX3090 1대: 0.765s</br>
RTX3090 4대: 0.238s</br>

# Preview (IPython을 사용한 gif변환)
![1](https://github.com/sjg918/mytetration/blob/main/gifs/out.gif?raw=true)
![2](https://github.com/sjg918/mytetration/blob/main/gifs/out1.gif?raw=true)
![3](https://github.com/sjg918/mytetration/blob/main/gifs/out2.gif?raw=true)
</br>

# 하고싶은일
잘 알려진 Fixed Point Arithmetic Library들에 따르면 정밀도를 높이기 위해 비트를 많이 사용할 수록 cpu와 gpu사이의 차이가 줄어듬</br>
그러니까 C FLINT Library를 사용한 eps 1e-13 이상 확대 구현 및 C++ 멀티 스레드 프로그램 구현</br>

# 원본출처
https://github.com/DMTPARK/mytetration</br>
https://youtu.be/pspWIwMFgws?si=Gdjm6SqXMrwFyECn</br>
