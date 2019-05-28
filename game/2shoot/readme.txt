



+사운드가 늦게 재생이 된다.  : fix
총알 소리, 폭발 소리가 한 박자 느리다.
wav 파일은 이상 없는 것 같다.
pygame.mixer.Sound( filename )
위 객체로 play() 했음.
==> FIX.
; pre_init으로 mixer 초기화. pygame.init 이전에 초기화. 버퍼 크기 줄임.


+무기 아이템.
아이템 먹을 때 마다 한 번에 발사되는 총알 증가. 중단, 상단, 하단.


+스코어.
텍스트 출력.
text align right로 하는 방법은??


+ bgm 재생 ; ok
mp3로 반복 재생
게임 종료시 잠깐 중단.
mixer.music.load(file)
mixer.music.play(-1)


+ 대기화면
game status에 따라 구분.
text 출력.


++ Spite 적용.
-group 별 처리가 용이함.
-충돌 체크가 편함. collide_mask 기능이 있어 정확한 충돌 감지 가능. (투명컬러 반영)
-프레임 변경시 이미지 갱신이 안된다? ; fix
이전 프레임을 지워줘야 한다. 자동이 아님... 투명 컬러로 sprite image에 fill로 삭제후 다시 그림.


+Enemy2
파괴 가능. 스코어 있음(파괴시 얻음).
공격기능없음. 충돌 체크 ; player, bullet

+ Enemy1
장애물 유형. 파괴되지 않음. 피해야 함.
충돌 체크 ; player
화면영역밖으로 가면 자동 self kill.

+ bullet
한 번 발사하면 일정 딜레이가 있음. 다중 발사 가능.
유형에 따라 발사 각도 설정.

+ player
상하좌우.
-키입력이 끊어짐 => FIX ; key.get_pressed() 사용.

+배경
하나의 이미지. 반복.
횡 스크롤.
