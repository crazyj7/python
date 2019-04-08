from bayes import BayesianFilter

bf = BayesianFilter()

#train
bf.fit("파격 세일 오늘까지만", "Spam")
bf.fit("쿠폰 선물 무료 배송", "Spam")
bf.fit("대리 운전 싸다", "Spam")
bf.fit("신제품 소식 봄과 함께 ", "Spam")
bf.fit("비아그라 남자 활력 지속", "Spam")
bf.fit("오늘 일정 확인", "Important")
bf.fit("프로젝트 진행 상황 보고", "Important")
bf.fit("회의 일정이 연기되었습니다.", "Important")
bf.fit("납품 기일은 다음 달 입니다.", "Important")

#predict
pre, scorelist = bf.predict("재고 정리 할인, 무료")
print(pre, scorelist)

pre, scorelist = bf.predict("회의 보고하세요")
print(pre, scorelist)

pre, scorelist = bf.predict("오늘 대리운전 공짜")
print(pre, scorelist)
