import numpy as np

# lst1 = np.array( [[1.2, 4], [5, 8], [19, 21], [24.5, 26]] )
# lst2 = np.array( [[1, 3], [6.55, 14.871], [22, 23]] )
#
# print(lst1)
# print(lst2)
#
# for r1 in lst1:
#     for r2 in lst2:
#         ri = np.intersect1d(r1, r2)
# intersect는 교집합.
#         print(ri)

# 범위인 경우?
# lst1 = [(1.2, 4), (5, 8), (19, 21), (24.5, 26)]
# lst2 = [(1, 3), (6.55, 14.871), (22, 23)]

lst1 = [(1.2, 4), (5, 8), (19, 21), (24.5, 26)]
lst2 = [(1, 3), (6.55, 14.871), (22, 25)]


def get_intersect(r1, r2):
    left = max(r1[0], r2[0])
    right = min(r1[1], r2[1])
    if left>right:
        return None
    return (left,right)

for i1 in lst1:
    for i2 in lst2:
        ia = get_intersect(i1, i2)
        if ia!=None:
            print(ia)

# but... merge??
# lst1 = [(5,24)]
#
# for i1 in lst1:
#     for i2 in lst2:
#         ia = get_intersect(i1, i2)
#         if ia!=None:
#             print(ia)
#
