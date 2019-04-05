# from konlpy.tag import Twitter
from konlpy.tag import Okt

#twitter = Twitter()
twitter = Okt()

print(twitter.morphs('단독입찰보다 복수입찰의 경우')) # 형태소 분리
print(twitter.nouns('단독입찰보다 복수입찰의 경우')) # 명사만
print(twitter.pos('이것도 되나요? ㅋㅋㅋ'))
print(twitter.pos('이것도 되나요? ㅋㅋㅋ', norm=True)) # 품사 표시 . ㅋㅋ도 변환
print(twitter.pos('이것도 되나요? ㅋㅋㅋ', norm=True, stem=True)) # 어근형태(동사원형)
