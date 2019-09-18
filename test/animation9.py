
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import math

ax1=None
ani=None

def ani_indextosegnum(vdraw, i):
    segindex=0
    totalcnt=0
    oldtotalcnt=0
    for s in vdraw:
        totalcnt += len(s)
        if i < totalcnt :
            break
        segindex+=1
        oldtotalcnt = totalcnt
        if segindex >= len(vdraw):
            return (-1,-1)
        if totalcnt < i :
            return (-1,-1)
    return (segindex, i-oldtotalcnt)

def ani_draw_animate(i, col, vrdraw):
   vdraw = vrdraw.params['draw']
   print('i=',i)
   segindex, newi = ani_indextosegnum(vdraw, i)
   if segindex<0:
      print('end')
      # 애니매이션 중지 2가지 방법: ani.event_source.stop()   # or use frame. norepeat
      return
   print('segindex=',segindex, 'newi=',newi)
   seg=vdraw[segindex]
   if len(seg) <= newi+1:
      ani.event_source.interval = 10
   else:
      ani.event_source.interval = seg[newi+1, 2]- seg[newi, 2]
   return ax1.plot( seg[0:newi, 0], seg[0:newi, 1], c=col, lw=3)

def ani_draw(vrdraw):
    global ax1, ani
   # vrdraw = Secudraw()
   # vrdraw.parse_drawdata(rdraw)
   # vrdraw.analyze()
    fig = plt.figure()
    ax1=fig.add_subplot(1,1,1)
    print(vrdraw.params['rmax'])
    print(vrdraw.params['rmin'])

    margin=20
    ax1.set_xlim(vrdraw.params['rmin'][0]-margin, vrdraw.params['rmax'][0]+margin)
    ax1.set_ylim(vrdraw.params['rmin'][1]-margin, vrdraw.params['rmax'][1]+margin)
    myline, = ax1.plot([],[], '-', c='k', lw=3)
    ani=animation.FuncAnimation(fig, ani_draw_animate, 
                                frames=vrdraw.params['ptcnt'] ,fargs=('k', vrdraw,),
                            interval=50, repeat=False,  blit=False)
    plt.show()
