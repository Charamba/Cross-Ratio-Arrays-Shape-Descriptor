from Image import *
from ShapeDescriptor import *

import itertools


class Scanner:
    def __init__(self, image):
        self.image = image

    def isValidBorderPoints(self, x, y):
        (height, width) = self.image.getShape()
        halfHeight = math.floor(height/2)
        halfWidth = math.floor(width/2)
        return (-halfWidth <= x <= halfWidth+1) and (-halfHeight <= y <= halfHeight+1)

    def calcImageBorderPoints(self, s, cosT, sinT):
        (height, width) = self.image.getShape()
        numberOfWhitePixels = 0
        halfHeight = math.floor(height/2) #float(height/2.0)#math.floor(height/2)
        halfWidth = math.floor(width/2)#float(width/2.0)#math.floor(width/2)#math.floor(width/2)

        borderPoints = []
        visitedPixels = []
        if cosT != 0:
            for y in [-halfHeight, halfHeight]:
                x = calcXbyY(s, cosT, sinT, y)
                if self.isValidBorderPoints(x, y):
                    borderPoints.append(R2_Point(x+halfWidth, y+halfHeight))
                    visitedPixels.append(R2_Point(math.floor(x+halfWidth), math.floor(y+halfHeight)))
        if sinT != 0:
            for x in [-halfWidth, halfWidth]:
                y = calcYbyX(s, cosT, sinT, x)
                if self.isValidBorderPoints(x, y):
                    borderPoints.append(R2_Point(x+halfWidth, y+halfHeight))
                    visitedPixels.append(R2_Point(math.floor(x+halfWidth), math.floor(y+halfHeight)))

        return (visitedPixels, borderPoints)

    def tomographic_scan(self, nTraj, nProj, calcCrossRatio=True, showTrajectories=False):
        (height, width) = self.image.getShape()
        # Initialize the transformed image
        
        diagonal = math.sqrt(height*height + width*width)
        dTheta = np.pi/nProj #3.14159265/nProj
        halfWidth = width/2
        halfHeight = height/2
        halfD = diagonal/2 # meia diagonal da imagem

        tMax = nProj
        sMax = diagonal/2
        ds = diagonal/(nTraj+1)

        descriptor = ShapeDescriptor()
        
        for t in range(0, tMax):
            theta = t*dTheta
            cosT = math.cos(theta)
            sinT = math.sin(theta)
            for sIdx in range(0, nTraj+1): #np.arange(0, sMax + ds, ds):
                s = sIdx*ds - sMax

                X = []
                Y = []
                firstPoint = True
                #print('s, t = %f, %f' %(s, t))
                
                edgePoints = []


                (visitedPixels__, borderPoints) = self.calcImageBorderPoints(s, cosT, sinT)
                if len(borderPoints) == 2:
                    P0 = borderPoints[0]
                    Pf = borderPoints[1]

                    # P0.x = math.floor(P0.x)
                    # P0.y = math.floor(P0.y)
                    # Pf.x = math.floor(Pf.x)
                    # Pf.y = math.floor(Pf.y)


                    (whitePixels, edgePoints, new_s) = self.image.calc_edgePoints(P0, Pf, s, theta, showTrajectories=showTrajectories)
                    #edgePoints = list(set(edgePoints))
                    #edgePoints = bresenham_line(s, cosT, sinT, x1, y1, x2, y2, image)
                    
                    
                    ##edgePoints = bresenham(x1, y1, x2, y2, image) # PAREI AQUI!

                    # if len(edgePoints) % 2 != 0:
                    #   ray = RayDescriptor(s, theta, edgePoints)
                    #   self.image.plotRay(ray)

                    descriptor.addRay(s, theta, edgePoints, whitePixels=whitePixels, calcCrossRatio=calcCrossRatio)

        return descriptor

    def fanBeam_deprecated(self, shapeDescriptor, P0, otherPoints, showTrajectories=False):
        (h, w) = self.image.getShape()

        for s, Pf in enumerate(otherPoints):
            (x0, y0) = P0.toTuple()
            (xf, yf) = Pf.toTuple()

            P0_ = R2_Point(x0 + w/2.0, y0 + h/2.0)
            Pf_ = R2_Point(xf + w/2.0, yf + h/2.0)

            P0_p2 = P0_.toP2_Point()
            Pf_p2 = Pf_.toP2_Point()

            R_p2 = P0_p2.cross(Pf_p2)
            R_p2.normalize()

            Pf__ = P0_ + 1*(Pf_ - P0_)
            (whitePixels, edgePoints, _) = self.image.calc_edgePoints(P0_, Pf__, showTrajectories=showTrajectories)
            

            # if edgePoints:
            #     if P0 != edgePoints[0]:
            #         edgePoints = [P0] + edgePoints 
                # if Pf != edgePoints[-1]:
                #     edgePoints = edgePoints + [Pf]
            theta = 0 ##  Fazer method para calcular theta
            #s = 1
            shapeDescriptor.addRay(s, theta, edgePoints, whitePixels=[])#whitePixels=whitePixels)

    def fanBeam(self, shapeDescriptor, P0, flag_inlcude_P0, otherPoints, showTrajectories=False, FULL_POINTS=False):
        (h, w) = self.image.getShape()
        rayList = []

        (height, width) = self.image.getShape()
        halfHeight = math.floor(height/2)
        halfWidth = math.floor(width/2)
        for s, Pf in enumerate(otherPoints):
            (x0, y0) = P0.toTuple()
            (xf, yf) = Pf.toTuple()

            P0_ = R2_Point(x0 + w/2.0, y0 + h/2.0)
            Pf_ = R2_Point(xf + w/2.0, yf + h/2.0)

            P0_p2 = P0_.toP2_Point()
            Pf_p2 = Pf_.toP2_Point()

            R_p2 = P0_p2.cross(Pf_p2)

            if R_p2.z == 0:
                continue

            R_p2.normalize()

            Pf__ = P0_ + 1*(Pf_ - P0_)
            #P0__ = P0_ - 0.1*(Pf_ - P0_)                             Pf__, P0_,
            (whitePixels, edgePoints, _) = self.image.calc_edgePoints(Pf__, P0_, showTrajectories=showTrajectories, FULL_POINTS=FULL_POINTS)
            
            P0_ = R2_Point(x0, y0)
            edgePoints.append(P0_)
            edgePoints = list(set(edgePoints))
            edgePoints = sortPoints(edgePoints, P0_)

            # Retirando primeiro ponto/vertice, caso não intercepte o objeto
            if not(flag_inlcude_P0):
                edgePoints = edgePoints[1:]

            # if len(edgePoints) % 2 != 0:
            #     edgePoints = edgePoints[1:]

            # if edgePoints:
            #     if P0 != edgePoints[0]:
            #         edgePoints = [P0] + edgePoints 
                # if Pf != edgePoints[-1]:
                #     edgePoints = edgePoints + [Pf]
            theta = 0 ##  Fazer method para calcular theta
            #s = 1
            xm = (x0 + xf)/2
            ym = (y0 + yf)/2
            #self.image.plotText(xm+halfWidth, ym+halfHeight, "s="+str(s), fontsize=12)

            ray = RayDescriptor(s, theta, edgePoints, whitePixels=[])
            rayList.append(ray)
            shapeDescriptor.addRay(s, theta, edgePoints, whitePixels=[]) # gambi para exibir SCAN RAYS
            # if s == 16 or s == 13:
            #     print("ray s=", s)
            #     print("n=", ray.numberOfEdgePoints)
            #     print(ray.edgePoints)
            #     print(ray.crossRatioVector)


        #rayList = sorted(templateRays, key=lambda r: r.s)
        shapeDescriptor.addPencil(rayList)

    def sampleTargetEdge(self, beamIndex, fanBeamRays, polygonVertices, edgeSizes):
        (h, w) = self.image.getShape()
        totalEdgeLen = sum(edgeSizes)
        #print("fanBeamRays = ", fanBeamRays)
        #for beamIndex in range(0, nVertices):


        sortedEdgeIndices = []

        nextIdx = beamIndex + 1 
        prevIdx = beamIndex - 1

        if prevIdx == -1:
            prevIdx = len(polygonVertices) - 1

        if nextIdx == len(polygonVertices):
            nextIdx = 0


        # print("totalEdgeLen = ", totalEdgeLen)
        # print("edgeSizes[beamIndex] = ", edgeSizes[beamIndex])
        # print("edgeSizes[prevIdx] = ", edgeSizes[prevIdx])

        edgesTargetSize = totalEdgeLen - (edgeSizes[beamIndex] + edgeSizes[prevIdx])
        step = edgesTargetSize/fanBeamRays

        if beamIndex == 0:
            # caso trivial
            #print("caso trivial")
            sortedEdgeIndices = range(nextIdx, prevIdx+1)
            #print(sortedEdgeIndices)
        elif beamIndex == 1:
            # caso especial 1
            #print("caso especial 1")
            sortedEdgeIndices = list(range(nextIdx, len(polygonVertices)))
            sortedEdgeIndices += [0]
            #print(sortedEdgeIndices)
        elif beamIndex == len(polygonVertices) - 1:
            # caso especial 2
            #print("caso especial 2")
            sortedEdgeIndices = range(0, prevIdx)
            #print(sortedEdgeIndices)
        else:
            # caso geral
            firstPart = list(range(nextIdx, len(polygonVertices)))
            secondPart = list(range(0, prevIdx+1))
            # print("caso geral")
            # print("firstPart = ", firstPart)
            # print("secondPart = ", secondPart)
            sortedEdgeIndices = firstPart + secondPart

        targetPoints = []
        for i in range(0, len(sortedEdgeIndices)):
            if i+1 < len(sortedEdgeIndices):
                eIdx0 = sortedEdgeIndices[i]
                eIdxf = sortedEdgeIndices[i+1]
                pv0 = polygonVertices[eIdx0]
                pvf = polygonVertices[eIdxf]

                (x0, y0) = pv0.toTuple()
                (xf, yf) = pvf.toTuple()
                pv0_ = R2_Point(x0 + w/2.0, y0 + h/2.0)
                pvf_ = R2_Point(xf + w/2.0, yf + h/2.0)
                vf0 = pvf_ - pv0_
                edgeLen = vf0.length()
                vf0.r2Normalize()

                #step = edgeLen/fanBeamRays
                edgePoints = []
                # Amostrando pontos em uma aresta do invólucro convexo
                for l in np.arange(0, edgeLen, step):
                    Pt = pv0 + l*vf0
                    targetPoints.append(Pt)

        return targetPoints

    def hull_scan(self, polygonVertices, convexHullVertices_original, fanBeamRays=2, showTrajectories=False, verticeIndexList=[], FULL_POINTS=False):
        descriptor = ShapeDescriptor()
        descriptor.hullVertices = polygonVertices
        (h, w) = self.image.getShape()
        #print("DEBUG --- descriptor: ", descriptor)

        edgeSizes = []
        nVertices = len(polygonVertices)

        for i in range(0, nVertices):#range(-1, nVertices):
            if i + 1 < nVertices:
                pi = polygonVertices[i]
                piNext = polygonVertices[i+1]
                ve = piNext - pi
                edgeSizes.append(ve.length())

        pi = polygonVertices[-1]
        piNext = polygonVertices[0]
        ve = piNext - pi
        edgeSizes.append(ve.length())

        totalEdgeLen = sum(edgeSizes)

        if verticeIndexList == []:
            verticeIndexList = list(range(0, nVertices))


        # 1 Percorrer vértices e amostrar os pontos de anteparo iguais ao número de fanBeamRays dado
        # 2 Chamar fanBeam
        for iBeam in range(0, nVertices):
            if iBeam in verticeIndexList:
                P0 = polygonVertices[iBeam]
                targetPoints = self.sampleTargetEdge(iBeam, fanBeamRays, polygonVertices, edgeSizes)
                #print("i = %d, targetPoints = %d" %(iBeam, len(targetPoints)))
                flag_inlcude_P0 = P0 in convexHullVertices_original

                self.fanBeam(descriptor, P0, flag_inlcude_P0, targetPoints, showTrajectories=showTrajectories, FULL_POINTS=FULL_POINTS)
        return descriptor

    def hull_scan_for_contour(self, polygonVertices, convexHullVertices_original, targetPoints, fanBeamRays=2, showTrajectories=False, verticeIndexList=[]):
        descriptor = ShapeDescriptor()
        descriptor.hullVertices = polygonVertices
        (h, w) = self.image.getShape()
        #print("DEBUG --- descriptor: ", descriptor)

        edgeSizes = []
        nVertices = len(polygonVertices)

        for i in range(0, nVertices):#range(-1, nVertices):
            if i + 1 < nVertices:
                pi = polygonVertices[i]
                piNext = polygonVertices[i+1]
                ve = piNext - pi
                edgeSizes.append(ve.length())

        pi = polygonVertices[-1]
        piNext = polygonVertices[0]
        ve = piNext - pi
        edgeSizes.append(ve.length())

        totalEdgeLen = sum(edgeSizes)

        if verticeIndexList == []:
            verticeIndexList = list(range(0, nVertices))


        # 1 Percorrer vértices e amostrar os pontos de anteparo iguais ao número de fanBeamRays dado
        # 2 Chamar fanBeam
        for iBeam in range(0, nVertices):
            if iBeam in verticeIndexList:
                P0 = polygonVertices[iBeam]
                #targetPoints = self.sampleTargetEdge(iBeam, fanBeamRays, polygonVertices, edgeSizes)
                #print("i = %d, targetPoints = %d" %(iBeam, len(targetPoints)))
                flag_inlcude_P0 = P0 in convexHullVertices_original

                self.fanBeam(descriptor, P0, flag_inlcude_P0, targetPoints, showTrajectories=showTrajectories)
        return descriptor

    def hull_scanAntigo(self, polygonVertices, fanBeamRays=2, showTrajectories=False, verticeIndexList=[]):
        descriptor = ShapeDescriptor()
        (h, w) = self.image.getShape()
        print("DEBUG --- descriptor: ", descriptor)

        # Convex hull total length
        totalHullLen = 0 
        for i in range(-1, len(polygonVertices)):
            if i+1 < len(polygonVertices): 
                P0 = polygonVertices[i]
                Pf = polygonVertices[i+1]
                vf0 = Pf - P0
                totalHullLen += vf0.length()

        edges = {}
        finalPoints = []
        step = totalHullLen/fanBeamRays
        # Amostrando pontos nas arestas do polígono
        for i in range(-1, len(polygonVertices)):
            if i+1 < len(polygonVertices):
                # Pegando pontos (vizinhos) dois a dois
                P0 = polygonVertices[i]
                Pf = polygonVertices[i+1]
                (x0, y0) = P0.toTuple()
                (xf, yf) = Pf.toTuple()
                P0_ = R2_Point(x0 + w/2.0, y0 + h/2.0)
                Pf_ = R2_Point(xf + w/2.0, yf + h/2.0)
                vf0 = Pf_ - P0_
                edgeLen = vf0.length()
                vf0.r2Normalize()

                #step = edgeLen/fanBeamRays
                edgePoints = []
                # Amostrando pontos em uma aresta do invólucro convexo
                for l in np.arange(0, edgeLen, step):
                    Pl = P0 + l*vf0
                    edgePoints.append(Pl)
                    finalPoints.append(Pl)

                actualIdx = i
                nextIdx = i+1

                # tratamento circular 
                if i == -1:
                    actualIdx = len(polygonVertices) - 1
                if i+1 == len(polygonVertices):
                    nextIdx = 0
                edges[actualIdx] = edgePoints

        if verticeIndexList == []:
            verticeIndexList = range(0, len(polygonVertices))

        # Gerando raios através do fan-beam
        for iBeam, P0 in enumerate(polygonVertices):
            if iBeam in verticeIndexList:

                # Edge índices vizinhos
                prevIdx = iBeam - 1
                nextIdx = iBeam + 1

                if prevIdx == -1:
                    prevIdx = len(polygonVertices) - 1

                if nextIdx == len(polygonVertices):
                    nextIdx = 0

                sortedEdgeIndices = []

                if iBeam == 0:
                    # caso trivial
                    print("caso trivial")
                    sortedEdgeIndices = range(nextIdx, prevIdx)
                    print(sortedEdgeIndices)
                elif iBeam == 1:
                    # caso especial 1
                    print("caso especial 1")
                    sortedEdgeIndices = range(nextIdx, len(polygonVertices))
                    print(sortedEdgeIndices)
                elif iBeam == len(polygonVertices) - 1:
                    # caso especial 2
                    print("caso especial 2")
                    sortedEdgeIndices = range(0, nextIdx)
                    print(sortedEdgeIndices)
                else:
                    # caso geral
                    firstPart = list(range(nextIdx, len(polygonVertices)))
                    secondPart = list(range(0, prevIdx))
                    print("caso geral")
                    print("firstPart = ", firstPart)
                    print("secondPart = ", secondPart)
                    sortedEdgeIndices = firstPart + secondPart

                #edgeBeamIndices = []
                finalPoints = []
                for eIdx in sortedEdgeIndices:
                    finalPoints += edges[eIdx]

                self.fanBeam(descriptor, P0, finalPoints, showTrajectories=showTrajectories)
        return descriptor

    def hull_scan_antigo(self, polygonVertices, fanBeamRays=2, showTrajectories=False, verticeIndexList=[]):
        descriptor = ShapeDescriptor()
        (h, w) = self.image.getShape()
        #borderPoints = []

        #hullEdges = []

        # mPoint = R2_Point(0,0)
        # for vertex in polygonVertices:
        #     mPoint += vertex

        # mPoint = mPoint/(len(polygonVertices))

        if verticeIndexList == []:
            verticeIndexList = range(0, len(polygonVertices))
        print("verticeIndexList = ", verticeIndexList)

        finalPoints = []
        #bulkheadPoints = {}
        edges = {}
        for i in range(-1, len(polygonVertices)):
            if i+1 < len(polygonVertices): #and i in verticeIndexList:
                #print("i = ", i)
                P0 = polygonVertices[i]
                Pf = polygonVertices[i+1]
                (x0, y0) = P0.toTuple()
                (xf, yf) = Pf.toTuple()
                P0_ = R2_Point(x0 + w/2.0, y0 + h/2.0)
                Pf_ = R2_Point(xf + w/2.0, yf + h/2.0)
                vf0 = Pf_ - P0_
                edgeLen = vf0.length()
                vf0.r2Normalize()
                step = edgeLen/fanBeamRays
                edgePoints = []
                # Amostrando pontos numa aresta do invólucro convexo
                for l in np.arange(0, edgeLen, step):
                    Pl = P0 + l*vf0
                    edgePoints.append(Pl)
                    finalPoints.append(Pl)

                actualIdx = i
                nextIdx = i+1

                # circular
                if i == -1:
                    actualIdx = len(polygonVertices) - 1

                if i+1 == len(polygonVertices):
                    nextIdx = 0


                edges[actualIdx] = edgePoints
                #bulkheadPoints[(P0, Pf)] = intermediatePoints

        for iBeam, P0 in enumerate(polygonVertices):
            if iBeam in verticeIndexList:

                # Edge indices vizinhos
                prevIdx = iBeam - 1
                nextIdx = iBeam + 1

                if prevIdx == -1:
                    prevIdx = len(polygonVertices) - 1

                if nextIdx == len(polygonVertices):
                    nextIdx = 0

                sortedEdgeIndices = []

                if iBeam == 0:
                    # caso trivial
                    sortedEdgeIndices = range(nextIdx, prevIdx)
                elif iBeam == 1:
                    # caso especial 1
                    sortedEdgeIndices = range(nextIdx, len(polygonVertices))
                elif iBeam == len(polygonVertices) - 1:
                    # caso especial 2
                    sortedEdgeIndices = range(0, nextIdx)
                else:
                    # caso geral
                    firstPart = list(range(0, prevIdx))
                    secondPart = list(range(nextIdx, len(polygonVertices)))
                    print("firstPart = ", firstPart)
                    print("secondPart = ", secondPart)
                    sortedEdgeIndices =  firstPart + secondPart

                #edgeBeamIndices = []
                finalPoints = []
                for eIdx in sortedEdgeIndices:
                    finalPoints += edges[eIdx]

                self.fanBeam_deprecated(descriptor, P0, finalPoints, showTrajectories=showTrajectories)
                #for (extremePoints, intermediatePoints) in bulkheadPoints.items():
                    #if not P0 in extremePoints:
                       #self.fanBeam(descriptor, P0, intermediatePoints, showTrajectories=showTrajectories)
                #    break
                # break      
        ##self.image.plotLinePoints(finalPoints, color="x", writeOrder=True)
        # for i in range(-1, len(polygonVertices)):
        #     if i+1 < len(polygonVertices):
        #         Pa = polygonVertices[i-1]
        #         P0 = polygonVertices[i]
        #         Pf = polygonVertices[i+1]

        #         (xa, ya) = Pa.toTuple()
        #         (x0, y0) = P0.toTuple()
        #         (xf, yf) = Pf.toTuple()

        #         Pa_ = R2_Point(xa + w/2.0, ya + h/2.0)
        #         P0_ = R2_Point(x0 + w/2.0, y0 + h/2.0)
        #         Pf_ = R2_Point(xf + w/2.0, yf + h/2.0)

        #         va = Pa - P0
        #         vf = Pf - P0
        #         vaf = Pf - Pa
        #         vaf_length = vaf.length()
        #         step = vaf_length/fanBeamRays
        #         vaf.r2Normalize()
        #         #vr = vf + va

        #         #vt = mPoint - P0_
        #         #Pr = P0_ + vr

        #         bulkheadPoints = [] # bulkhead points
        #         for l in np.arange(0, vaf_length, step):
        #             Pl = Pa + l*vaf
        #             otherPoints.append(Pl)

        #         # step = (2.0*math.pi)/fanBeamRays
        #         # #vr = va + vf
        #         # for t in np.arange(0, 2.0, step):
        #         #     theta = t#math.pi*t
        #         #     cosT = math.cos(theta)
        #         #     sinT = math.sin(theta)
                    
        #         #     (x, y) = vt.toTuple()
        #         #     (vt.x, vt.y) = (x*cosT - y*sinT, x*sinT + y*cosT)
        #         #     Pt = P0_ + vt
        #         #     otherPoints.append(Pt)

        #         self.fanBeam(descriptor, P0, otherPoints)
                # edge = P0_.cross(Pf_)
                # edge.normalize()
                # hullEdges.append(edge)

                # Pa
                #break # <==================

                #(_, borderPoints_) = self.image.calcPixelBorderPoints(P0_, Pf_)
                #borderPoints = borderPoints + borderPoints_


        #FAN BEAM
        #polygonVertices = [polygonVertices[0]]

        # if fanBeamRays >= 2:
        #     for P0 in polygonVertices:
        #         nSamples = fanBeamRays
        #         step = int(len(borderPoints)/nSamples)
        #         idx = range(0, len(borderPoints), step)
        #         sampledBorderPoints = [borderPoints[i] for i in idx]
        #         self.fanBeam(descriptor, P0, sampledBorderPoints)

        verticesCombinations = []
        # for i in range(0, len(polygon)):
        #     combinations += [(polygon[i],p2) for p2 in polygon[i+1:]]

        #combinations = pairsCombinations(polygon)
        verticesCombinations = []#itertools.combinations(polygonVertices, 2)
        
        for (P0, Pf) in verticesCombinations:
            (x0, y0) = P0.toTuple()
            (xf, yf) = Pf.toTuple()

            P0_ = R2_Point(x0 + w/2.0, y0 + h/2.0)
            Pf_ = R2_Point(xf + w/2.0, yf + h/2.0)
            (whiteVisitedPixels, edgePoints, s) = self.image.calc_edgePoints(P0_, Pf_, showTrajectories=showTrajectories)

            newEdges = []
            last = None

            # seen = set()
            # seen_add = seen.add
            # newEdges = [x for x in edgePoints if not (x in seen or seen_add(x))]

            # edgePoints = newEdges

            #edgePoints = list(set(edgePoints))

            if edgePoints:
                if P0 != edgePoints[0]:
                    edgePoints = [P0] + edgePoints 
                if Pf != edgePoints[-1]:
                    edgePoints = edgePoints + [Pf]
            theta = 0 ##  Fazer method para calcular theta

            descriptor.addRay(s, theta, edgePoints, whitePixels=whiteVisitedPixels)
        return descriptor
