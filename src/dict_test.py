#dict_test.py

from Point import *

class HoughLines:
    def __init__(self):
        self.lines = {}
    def getLines(self):
        return self.lines.items()
    def updateLine(self, key, point):
        if key in self.lines:
            self.lines[key].append(point)
        else:
            self.lines.update({key:[point]})

P0 = R2_Point(0, 20)
P1 = R2_Point(1, -2)
P2 = R2_Point(13, 6)
P3 = R2_Point(40, 5)
P4 = R2_Point(10, 9)


'''
lines = {(0,0): [P0, P1], (0,3): [P1]}
#print(lines)

lines[(0,0)].append(P3)
#print(lines)

for key, val in lines.items():
    if len(val) == 3:
        print(key, val)

lines2 = {}

if (0,1) in lines2:
    lines2[(0,1)].append(P1)
else:
    lines2.update({(0,1):[P1]})

print("lines2: ", lines2)

if (0,1) in lines2:
    lines2[(0,1)].append(P2)
else:
    lines2.update({(0,1):[P2]})

print("lines2: ", lines2)
'''

houghLines = HoughLines()
houghLines.updateLine((0,0), P0)
print(houghLines.getLines())
houghLines.updateLine((0,0), P1)
print(houghLines.getLines())
houghLines.updateLine((0,1), P2)
print(houghLines.getLines())
houghLines.updateLine((0,0), P3)
print(houghLines.getLines())
