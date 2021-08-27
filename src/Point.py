import numpy as np
import math

#from sets import Set

class P2_Point(object): #Point
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return P2_Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return P2_Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if(type(other) == P2_Point):
            return (self.x * other.x + self.y * other.y + self.z * other.z)
        else:
            return P2_Point(self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, other):
        if(type(other) == P2_Point):
            return (self.x * other.x + self.y * other.y + self.z * other.z)
        else:
            return P2_Point(self.x * other, self.y * other, self.z * other)
    
    def __truediv__(self, other):
        if (type(other) != int and type(other) != float and
            type(other) != np.int_ and type(other) != np.float_):
            raise Exception("Undefined P2_Point division")

        if other == 0:
            raise Exception("Division by zero")

        return P2_Point(self.x / other, self.y / other, self.z / other)
    
    def __str__(self):
        return "(%f, %f, %f)" % (self.x, self.y, self.z)

    def toTuple(self):
        return (self.x, self.y, self.z)


    def euclideanDistance(self,other):
        diff = self-other
        return math.sqrt(diff*diff)
    
    def cross(self, other):
        return P2_Point(self.y*other.z - self.z*other.y, 
                     self.z*other.x - self.x*other.z, 
                     self.x*other.y - self.y*other.x)

    def get_pixel_coord(self, xSize, ySize):
        self.normalize()
        x = int(round(self.x + (xSize / 2)))
        y = int(round((ySize / 2) - self.y))

        return (y, x)


    def to_img_coord(self, halfWidth, halfHeight):
        self.x = self.x + halfWidth
        self.y = self.y + halfHeight
        # self.x = self.x + (xSize / 2)
        # self.y = (ySize / 2) - self.y

    def normalize(self):
        self.x /= self.z
        self.y /= self.z
        self.z = 1

    def r3Normalize(self):
        modulos = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        self.x /= modulos
        self.y /= modulos
        self.z /= modulos


    def transform(self, H):
        if len(H) != 3:
            raise Exception("Incorrect matrix size")

        for line in H:
            if len(line) != 3: 
                raise Exception("Incorrect matrix size")

        new_point = np.array([ H[0][0]*self.x + H[0][1]*self.y + H[0][2]*self.z, 
                               H[1][0]*self.x + H[1][1]*self.y + H[1][2]*self.z, 
                               H[2][0]*self.x + H[2][1]*self.y + H[2][2]*self.z ])

        self.x = new_point[0]
        self.y = new_point[1]
        self.z = new_point[2]

    def to_nparray(self):
        return np.array([self.x, self.y, self.z])

    def toR2_Point(self):
        self.normalize()
        return R2_Point(self.x, self.y)

class R2_Point(object): #Point2
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return other and other.x -1E-5 <= self.x <= other.x + 1E-5 and other.y - 1E-5 <= self.y <= other.y + 1E-5
        #return other and self.x == other.x and self.y == other.y
    def __add__(self, other):
        return R2_Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return R2_Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if type(other) == R2_Point:
            return (self.x * other.x + self.y * other.y)
        else:
            return R2_Point(self.x * other, self.y * other)
    
    def __rmul__(self, other):
        if type(other) == R2_Point:
            return (self.x * other.x + self.y * other.y)
        else:
            return R2_Point(self.x * other, self.y * other)
    
    def __truediv__(self, other):
        if (type(other) != int and type(other) != float and
            type(other) != np.int_ and type(other) != np.float_):
            raise Exception("Undefined R2_Point division")

        if other == 0:
            raise Exception("Division by zero")

        return R2_Point(self.x / other, self.y / other)
    
    def __str__(self):
        return "(%f, %f)" % (self.x, self.y)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.x, self.y))

    def toTuple(self):
        return (self.x, self.y)

    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y)

    def euclideanDistance(self,other):
        diff = self-other
        return math.sqrt(diff*diff)

    def get_pixel_coord(self, xSize, ySize):
        self.normalize()
        x = int(round(self.x + (xSize / 2)))
        y = int(round((ySize / 2) - self.y))
        return (y, x)
    def to_default_coord(self, halfWidth, halfHeight):
        self.x = self.x - halfWidth
        self.y = self.y - halfHeight
    def to_cartesian_coord(self, xSize, ySize):
        self.x = self.x - (xSize / 2)
        self.y = (ySize / 2) - self.y

    def to_img_coord(self, halfWidth, halfHeight):
        self.x = self.x + halfWidth
        self.y = self.y + halfHeight

    def normalize(self):
        self.x /= self.y
        self.y = 1

    def r2Normalize(self):
        modulos = math.sqrt(self.x*self.x + self.y*self.y)
        self.x /= modulos
        self.y /= modulos

    def transform(self, H):
        if len(H) != 2:
            raise Exception("Incorrect matrix size 1 ")

        for line in H:
            if len(line) != 2: 
                raise Exception("Incorrect matrix size 2")

        new_point = np.array([ H[0][0]*self.x + H[0][1]*self.y,  
                               H[1][0]*self.x + H[1][1]*self.y ])

        self.x = new_point[0]
        self.y = new_point[1]

    def to_nparray(self):
        return np.array([self.x, self.y])

    def toP2_Point(self):
        return P2_Point(self.x, self.y, 1)

class WeightedPoint(R2_Point):
    def __init__(self, x, y, w):
       R2_Point.__init__(self, x, y)
       self.w = w
'''
print("OPA!")
P0 = R2_Point(1.0, 1.5)
P1 = R2_Point(1.0, 2.5)
P2 = R2_Point(1.0, 1.5)

points = []
points.append(P0)
points.append(P1)
points.append(P2)
pointsSet = set(points)
print("-----------------")
print("pointsSet = ", pointsSet)
'''
