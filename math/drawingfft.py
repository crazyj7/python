
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import splrep, splev


# pts = '[(162,225,0)(162,225,41)(157,223,59)(151,223,75)(143,222,92)(132,222,108)(119,223,125)(104,232,142)(92,245,158)(79,265,175)(71,281,191)(63,294,208)(60,303,226)(59,307,242)(59,308,259)(61,308,275)(76,297,292)(101,281,308)(133,260,325)(162,235,342)(189,208,359)(207,188,375)(222,168,392)(232,157,408)(237,149,425)(239,146,442)(240,145,457)(237,149,491)(219,166,508)(191,194,525)(162,224,543)(135,260,558)(111,298,575)(96,323,592)(89,340,609)(86,347,625)(86,348,642)(97,341,658)(124,315,675)(157,285,692)(192,255,708)(220,230,725)(236,217,742)(242,214,758)(243,213,775)(240,217,808)(228,232,825)(214,249,842)(201,269,859)(190,289,875)(180,309,892)(174,325,909)(172,335,925)(171,340,942)(171,341,959)(181,337,975)(204,323,992)(227,309,1008)(249,295,1025)(266,280,1041)(275,271,1058)(277,267,1075)(277,265,1092)(268,265,1108)(245,272,1125)(222,284,1141)(201,294,1158)(189,298,1175)(184,298,1192)(182,299,1208)(180,300,1223)]'

pts='[(202,85,0)(202,85,30)(196,89,48)(191,98,66)(183,114,83)(173,137,98)(162,162,115)(151,187,132)(141,212,149)(131,235,166)(121,256,183)(112,277,199)(105,294,214)(100,311,233)(95,327,249)(91,340,266)(88,352,283)(86,361,299)(85,367,316)(84,370,333)(84,371,348)(84,371,382)(84,366,399)(88,350,416)(97,317,433)(109,280,449)(121,243,466)(132,211,483)(145,183,498)(154,159,516)(163,136,533)(168,119,549)(172,107,566)(175,99,583)(179,94,599)(183,91,614)(188,90,630)(196,90,648)(203,90,664)(212,96,682)(221,105,699)(232,118,716)(249,135,732)(266,152,749)(285,172,766)(302,192,783)(317,210,799)(329,224,816)(338,234,832)(343,239,849)(346,244,866)(347,246,881)(347,246,898)(343,246,930)(329,246,949)(307,246,966)(282,244,982)(257,241,999)(233,240,1016)(211,240,1032)(190,240,1049)(174,240,1065)(161,240,1082)(151,240,1099)(143,240,1116)(137,240,1132)(134,241,1149)(134,241,1166)(134,241,1182)(134,242,1199)(134,242,1216)(139,243,1232)(148,245,1249)(165,245,1266)(191,245,1282)(215,240,1299)(233,234,1315)(250,228,1332)(260,223,1349)(266,219,1366)(269,217,1382)(269,216,1399)(269,214,1416)(265,213,1432)(256,212,1449)(247,211,1466)(239,211,1482)(231,211,1499)(224,215,1516)(219,221,1532)(215,228,1549)(211,240,1565)(208,256,1582)(205,277,1599)(202,298,1616)(202,315,1632)(202,335,1649)(202,354,1666)(202,372,1681)(203,385,1697)(203,400,1713)(203,413,1729)(204,425,1746)(205,436,1764)(206,443,1780)(206,448,1797)(207,450,1814)(207,450,1831)(207,450,1849)(207,441,1866)(207,418,1882)(207,400,1899)(207,382,1916)(207,372,1932)(205,364,1949)(203,358,1966)(199,354,1982)(193,350,1999)(188,348,2015)(181,348,2032)(173,348,2049)(167,348,2065)(160,349,2082)(154,351,2099)(151,353,2115)(151,353,2132)(151,354,2149)(156,354,2166)(172,352,2182)(194,348,2199)(216,342,2216)(231,338,2232)(243,334,2249)(251,332,2266)(256,331,2282)(259,330,2299)(260,330,2316)(260,330,2332)(260,330,2349)(252,333,2366)(240,340,2383)(219,353,2399)(195,368,2416)(172,384,2432)(152,397,2449)(138,406,2466)(129,412,2482)(124,416,2499)(122,418,2516)(122,420,2532)(123,421,2549)(129,421,2565)(141,422,2582)(161,422,2599)(188,423,2616)(225,423,2632)(259,422,2649)(286,420,2665)(306,419,2682)(317,418,2699)(324,418,2715)(326,418,2732)(327,418,2749)(327,417,2766)(327,416,2782)(328,413,2798)]'

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
sign = np.asarray(sign)[0]  # first drawing (1 count)

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


# print('sign=',sign)
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

def get_len(v1,v2):
    partlen = np.sqrt((v2[0] - v1[0])**2 + (v2[1]-v1[1])**2)
    return partlen


def get_length(vec):
    '''
    get the length of the line points
    :param vec: [x,y,..][x,y,..]
    :return: length
    '''
    if vec.ndim>1:
        startx = vec[:-1,0]
        endx = vec[1:,0]
        starty = vec[:-1,1]
        endy = vec[1:,1]
    else:
        startx = vec[:-1]
        endx = vec[1:]
        starty = np.zeros(len(vec)-1)
        endy = starty
    # print('cnt=', len(startx), len(endx))
    partlen = np.sqrt((endx - startx)**2 + (endy-starty)**2)
    return np.sum(partlen)


N = 1000    # sample count (extended)
fs = 1000   # sampling frequency
T = 1/fs

def my_fft(orgt, val, extracnt):
    global pi
    spl = splrep(orgt, val)
    newt = np.linspace(0,1,N)
    newval = splev(newt, spl)

    # print('newval=', newval)
    # 전체 드로잉 길이
    totallen = get_length(newval)
    # 포인트당 평균 이동 길이
    lenpt = totallen/N
    print('totallen=', totallen, 'len/ptcnt=', lenpt)

    ## gibb's pheonomenum problem...
    # remove first and last some part??? and save it aside?...
    # or add extra link from end point to start point.
    #  remove extra time.
    lastt = newt[-1]
    lastv = newval[-1]
    dt = newt[1]-newt[0]
    endtostartlen = newval[0] - newval[-1]
    dl = endtostartlen / extracnt
    print('lastv=', lastv, 'startv=', newval[0])
    print('newval cnt=', len(newval))
    print('extracnt = ', extracnt, 'dt=',dt,'dl=', dl)
    addt = []
    addv = []
    for i in range(extracnt):
        addt.append(lastt+(i+1)*dt)
        addv.append(lastv+(i+1)*dl)
    addt = np.asarray(addt)
    addv = np.asarray(addv)
    # print(addt)
    # print(addv)
    newt = np.r_[newt, addt]
    newval = np.r_[newval, addv]
    print(newt.shape, newval.shape)

    # same shape. but more detail... sampling..
    if False:
        pi+=1
        plt.subplot(ph, pw, pi)
        plt.plot(newt, newval)

    # fft
    nfft=N+extracnt  # 2'power but. python process this in backend automatically. so no need.
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
    newy = np.zeros((nfft,))
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

    return newt, newy, newval, freq, coec, coes


# t, val
shapelen = get_length(sign)
print('shapelen=', shapelen)
dl = shapelen/N
endtostartlen = get_len([sign[-1,0], sign[-1,1]], [sign[0,0], sign[0,1]])
print('endtostartlen=', endtostartlen)
extracnt = int(endtostartlen/dl)
print('extracnt=', extracnt)


newt, newx, oldx, freq, coec, coes = my_fft(sign[:,2], sign[:,0], extracnt)
newt, newy, oldy, freq, coec, coes = my_fft(sign[:,2], sign[:,1], extracnt)

pi+=1
plt.subplot(ph, pw, pi)
plt.plot(sign[:,0], sign[:,1])


pi+=1
plt.subplot(ph, pw, pi)
plt.plot(newx[:-extracnt], newy[:-extracnt])
# plt.scatter(newx[0], newy[0], c='r')
# plt.scatter(newx[-1], newy[-1], c='b')

plt.show()





