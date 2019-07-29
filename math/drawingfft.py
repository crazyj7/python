
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import splrep, splev


pts = '[(162,225,0)(162,225,41)(157,223,59)(151,223,75)(143,222,92)(132,222,108)(119,223,125)(104,232,142)(92,245,158)(79,265,175)(71,281,191)(63,294,208)(60,303,226)(59,307,242)(59,308,259)(61,308,275)(76,297,292)(101,281,308)(133,260,325)(162,235,342)(189,208,359)(207,188,375)(222,168,392)(232,157,408)(237,149,425)(239,146,442)(240,145,457)(237,149,491)(219,166,508)(191,194,525)(162,224,543)(135,260,558)(111,298,575)(96,323,592)(89,340,609)(86,347,625)(86,348,642)(97,341,658)(124,315,675)(157,285,692)(192,255,708)(220,230,725)(236,217,742)(242,214,758)(243,213,775)(240,217,808)(228,232,825)(214,249,842)(201,269,859)(190,289,875)(180,309,892)(174,325,909)(172,335,925)(171,340,942)(171,341,959)(181,337,975)(204,323,992)(227,309,1008)(249,295,1025)(266,280,1041)(275,271,1058)(277,267,1075)(277,265,1092)(268,265,1108)(245,272,1125)(222,284,1141)(201,294,1158)(189,298,1175)(184,298,1192)(182,299,1208)(180,300,1223)]'


def ss_reverseY(sign):
    '''reverse Y axis.
    '''
    matmax = []
    matmin = []
    for seg in sign:
        matmax.append(np.max(seg, 0))
        matmin.append(np.min(seg, 0))
    rmax = np.max(matmax, 0)
    rmin = np.min(matmin, 0)
    for seg in sign:
        # y axis reverse!
        seg[:, 1] = rmax[1] - seg[:, 1] + rmin[1]
    return sign



# snsign : [(1,2,3)(4,5,6)(1,1,1)][(1,1,1)(2,2,2)]
def parse_signdata(snsign, breversey=True):
    '''parsing sign data
    input : SN type ; [segment(x,y,t),...][segment(x,y,t),..]..
    output: sign(list)
    '''
    snsign = snsign.rstrip()  # remove CR, space char
    sign = []
    tmp1 = snsign.replace('[', '')  # remove [ char
    tmp1 = tmp1.rstrip(']')  # remove last ] char
    tmp1 = tmp1.split(']')
    if type(tmp1) == str:
        tmp1 = [tmp1]
    for node in tmp1:
        node = node.replace('(', '')
        node = node.replace(')', ' ')
        node = node.replace(',', ' ')
        ml = np.fromstring(node, dtype=float, sep=' ').reshape(-1, 3)
        sign.append(ml)

    if len(sign) == 0 or (len(sign) == 1 and len(sign[0]) == 0):
        return None

    # reverse Y
    if breversey:
        sign = ss_reverseY(sign)

    return sign


sign = parse_signdata(pts)
sign = np.asarray(sign)[0]

signmin = np.min(sign, axis=0)
# zero-base. positive number
sign = sign-signmin
# normalzie.
signsize = np.max(sign, axis=0)
# x,y range [0,2]
sign[:,0] /= signsize[0]/2
sign[:,1] /= signsize[1]/2
sign[:,2] /= signsize[2]
# x,y range [-1,1]
sign[:,0] -= 1
sign[:,1] -= 1


print('sign=',sign)
print(sign.shape)


pi=1
ph=4
pw=3

plt.figure()
plt.subplot(ph,pw,pi)
plt.title('x-y')
plt.plot(sign[:,0], sign[:,1])

pi+=1
plt.subplot(ph,pw,pi)
plt.title('t-x')
plt.plot(sign[:,2], sign[:,0])

pi+=1
plt.subplot(ph,pw,pi)
plt.title('t-y')
plt.plot(sign[:,2], sign[:,1])



def my_fft(orgt, val):
    global pi
    N = 1000    # sample count (extended)
    fs = 1000   # sampling frequency
    T = 1/fs
    spl = splrep(orgt, val)
    newt = np.linspace(0,1,N)
    newval = splev(newt, spl)

    # same shape. but more detail... sampling..
    if False:
        pi+=1
        plt.subplot(ph, pw, pi)
        plt.plot(newt, newval)

    # fft
    nfft=N  # 2'power but. python process this in backend automatically. so no need.
    k = np.arange(nfft)
    f = k*fs/nfft      # frequency sample. (sample count)
    f0 = f[range(math.trunc(nfft/2))]      # only half size check for get hz.

    y = np.fft.fft(newval)/nfft * 2
    y0 = y[range(math.trunc(nfft/2))]       # one side.

    amp = abs(y0)
    phase = np.angle(y0)

    pi+=1
    plt.subplot(ph, pw, pi)
    plt.plot(f0, amp)
    pi+=1
    plt.subplot(ph, pw, pi)
    plt.plot(f0, phase)

    # get top N
    topn=10

    idxy = np.argsort(-amp)
    for i in range(topn):
        print('freq=', f0[idxy[i]], 'amp=', y[idxy[i]], 'angle=', phase[idxy[i]])

    # recover
    newy = np.zeros((N,))
    for i in range(topn):
        freq = f0[idxy[i]]
        yx = y[idxy[i]]
        coec = yx.real
        coes = yx.imag * -1
        ang = phase[idxy[i]]
        print('freq=', freq, 'coec=', coec, ' coes', coes, 'ang', ang)
        # newy += coec * np.cos( 2*np.pi*freq*newt+ang) + coes * np.sin( 2*np.pi*freq*newt+ang)
        newy += coec * np.cos(2 * np.pi * freq * newt ) + coes * np.sin(2 * np.pi * freq * newt)
    pi+=1
    plt.subplot(ph,pw,pi)
    plt.plot(newt, newval)
    plt.plot(newt, newy)

    ## gibb's pheonomenum problem...
    # remove first and last some part??? and save it aside?...

    return newt, newy, newval, freq, coec, coes


# t, val
newt, newx, oldx, freq, coec, coes = my_fft(sign[:,2], sign[:,0])
newt, newy, oldy, freq, coec, coes = my_fft(sign[:,2], sign[:,1])

pi+=1
plt.subplot(ph, pw, pi)
plt.plot(oldx, oldy)
plt.scatter(oldx[0], oldy[0], c='r')
plt.scatter(oldx[-1], oldy[-1], c='b')
plt.plot(newx, newy)
plt.scatter(newx[0], newy[0], c='r')
plt.scatter(newx[-1], newy[-1], c='b')

plt.show()





