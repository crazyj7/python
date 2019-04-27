유전자 알고리즘과 딥러닝 학습을 결합하여
구글 공룡 게임을 학습시킴.
피쳐에 대한 가중치값을 찾는 fiting을 강화학습이 아닌 유전자 알고리즘으로 찾는 듯.
 

# Genetic Dino

![result.gif](https://github.com/kairess/genetic_dino/raw/master/result.gif)

- Used neural networks for generating genomes. (input - 2 hidden layers - output, with tanh activations)
- Input values are distance from Dino to nearest obstacle and top of the nearest obstacle
- Thanks to Shivam Shekhar for sharing [T-Rex Rush](https://github.com/shivamshekhar/Chrome-T-Rex-Rush)

## Dependencies
- Python 3+
- pygame
- numpy
- matplotlib (for ploting)

## Run
```
python game.py
```