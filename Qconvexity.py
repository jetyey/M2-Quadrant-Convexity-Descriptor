import argparse
import cv2
import numpy as np

#Function that returns a matrix of Q-concavity values for each pixel
def nonQconvexity(gray):
    r, c =  gray.shape # r = rows/height and c = columns/width
    sumOfNonQConvexityMeasure = 0
    nonQConvexityMeasureMatrix = np.array([[0 for x in range(c)] for y in range(r)], dtype="float32")

    for i in range(r):
        for j in range(c):
            if(gray[i,j] == 255):
                f = 0
            else:
                f = 1
            nonQConvexityMeasureMatrix[i,j] = quandrant1(gray,i,j) * quandrant2(gray,i,j) * quandrant3(gray,i,j) * quandrant4(gray,i,j) *f
            sumOfNonQConvexityMeasure += nonQConvexityMeasureMatrix[i, j] 
    
    return nonQConvexityMeasureMatrix

#Function that returns the number of object points in img(i,j) for quadrant 1
def quandrant1(img, r, c):
    quad1 = 0

    for i in range(r+1):
        for j in range(c+1):
            if not(i==r and j==c):
                if(img[i,j] == 255):
                    quad1 += 1
    
    return quad1

#Function that returns the number of object points in img(i,j) for quadrant 2
def quandrant2(img, r, c):
    quad2 = 0
    row , col = img.shape

    for i in range(r+1):
        for j in range(c, col):
            if not(i==r and j==c):
                if(img[i,j] == 255):
                    quad2 += 1
    
    return quad2

#Function that returns the number of object points in img(i,j) for quadrant 3
def quandrant3(img, r, c):
    quad3 = 0
    row , col = img.shape

    for i in range(r, row):
        for j in range(c+1):
            if not (i==r and j==c):
                if(img[i,j] == 255):
                    quad3 += 1

    return quad3

#Function that returns the number of object points in img(i,j) for quadrant 4
def quandrant4(img, r, c):
    quad4 = 0
    row, col = img.shape

    for i in range(r, row):
        for j in range(c, col):
            if not (i==r and j==c):
                if(img[i,j] == 255):
                    quad4 += 1
    
    return quad4

#Function that returns the number of object foreground, an array of the number of object foreground for each row/column
def countObjectPixel(img):
    r, c = img.shape
    count = 0
    vertic = [[0 for x in range(c)] for y in range(1)]
    horiz = [[0 for x in range(r)] for y in range(1)]    

    for i in range(r):
        for j in range(c):
            if(img[i,j] == 255):
                vertic[0][j] += 1
                horiz[0][i] += 1
                count += 1

    return count, vertic, horiz

#Function that return an array of normalized value of Q-concavity matrix for each pixel using the global method
def globalMethod(matrixNonQConvexValues):
    r, c = matrixNonQConvexValues.shape
    matrixGlobalNormalize = np.array([[0.0 for x in range(c)] for y in range(r)], dtype="float32")
    maxVal  = np.amax(matrixNonQConvexValues)
    if(maxVal != 0):

        for i in range(r):
            for j in range(c):
                matrixGlobalNormalize[i,j] = (matrixNonQConvexValues[i,j]/maxVal)

        return matrixGlobalNormalize
    else:
        return matrixGlobalNormalize

#Function that return an array of normalized value of Q-concavity matrix for each pixel using the local method
def localMethod(matrixNonQConvexValues, vertic, horiz, numObjPixel):
    r, c = matrixNonQConvexValues.shape
    matrixLocalNormalize = np.array([[0.0 for x in range(c)] for y in range(r)], dtype="float32")

    for i in range(r):
        for j in range(c):
            matrixLocalNormalize[i,j] = (matrixNonQConvexValues[i,j]/(((numObjPixel+horiz[0][i]+vertic[0][j])/4)**4))

    return matrixLocalNormalize

#Function that returns values of Enlacement landscape for both local and global method
def enlacementLanscapeMeasure(matrixGlobalNormalize, matrixLocalNormalize):
    r,c = matrixGlobalNormalize.shape
    globalNonNull = np.count_nonzero(matrixGlobalNormalize)
    localNonNull = np.count_nonzero(matrixLocalNormalize)
    phiGlobal = 0.0
    phiLocal = 0.0
    for i in range(r):
        for j in range(c):
            if(globalNonNull != 0): phiGlobal += (matrixGlobalNormalize[i, j]/globalNonNull)
            if(localNonNull != 0): phiLocal += (matrixLocalNormalize[i, j]/localNonNull)
    
    return phiGlobal, phiLocal

#Function that returns an array containing either 0 or Q-concavity value
def spatialRelationEnlacement(matrixNonQConvexValues, img):
    r,c = matrixNonQConvexValues.shape
    matrixSpatialRelationEnlacement = np.array([[0.0 for x in range(c)] for y in range(r)], dtype="float32")

    for i in range(r):
        for j in range(c):
            if(img[i,j] > 0):
                matrixSpatialRelationEnlacement[i, j] = matrixNonQConvexValues[i, j]

    return matrixSpatialRelationEnlacement

#Function that returns enlacement descriptors values using global and local method for spatial relations
def spatialRelationEnlacementMeasure(msreGlobalMethod, msreLocalMethod, img):
    r, c = msreGlobalMethod.shape
    globalCount, localCount, phiGlobalSR, phiLocalSR = 0.0, 0.0, 0.0, 0.0

    for i in range(r):
        for j in range(c):
            if(img[i, j] == 255 and msreGlobalMethod[i, j] > 0): globalCount += 1.0
            if(img[i, j] == 255 and msreLocalMethod[i, j] > 0): localCount += 1.0

    for i in range(r):
        for j in range(c):
            if(globalCount != 0): phiGlobalSR += (msreGlobalMethod[i, j]/globalCount)
            if(localCount != 0): phiLocalSR += (msreLocalMethod[i, j]/localCount)
    
    return phiGlobalSR, phiLocalSR

#Function that returns interlacement values using global and local normalization methods
def calculateInterlacementMeasure(epsilonGlobalFG, epsilonLocalFG, epsilonGlobalGF, epsilonLocalGF):
    interlacementGlobal = (2 * epsilonGlobalFG * epsilonGlobalGF)/(epsilonGlobalFG + epsilonGlobalGF)
    interlacementLocal = (2 * epsilonLocalFG * epsilonLocalGF)/(epsilonLocalFG + epsilonLocalGF)

    return interlacementGlobal, interlacementLocal

#Function that returns enlacmeent of 1 object values using global and local normalization methods
def calculateEnlacementMeasureFrom1Object(imageF):
    numObjPixel,vertic, horiz = countObjectPixel(imageF)
    matrixNonQConvexValues = nonQconvexity(imageF)
    matrixSpatialRelationEnlacementGlobalMethod = globalMethod(matrixNonQConvexValues)
    matrixSpatialRelationEnlacementLocalMethod = localMethod(matrixNonQConvexValues, vertic, horiz, numObjPixel)
    epsilonGlobalFG, epsilonLocalFG = enlacementLanscapeMeasure(matrixSpatialRelationEnlacementGlobalMethod, matrixSpatialRelationEnlacementLocalMethod)

    return epsilonGlobalFG, epsilonLocalFG

#Function that returns enlacmeent of spatial relations between 2 objects values using global and local normalization methods
def calculateEnlacementMeasureFrom2Objects(imageF, imageG):
    numObjPixel,vertic, horiz = countObjectPixel(imageF)
    matrixNonQConvexValues = nonQconvexity(imageF)
    matrixSpatialRelationEnlacement = spatialRelationEnlacement(matrixNonQConvexValues, imageG)
    matrixSpatialRelationEnlacementGlobalMethod = globalMethod(matrixSpatialRelationEnlacement)
    matrixSpatialRelationEnlacementLocalMethod = localMethod(matrixSpatialRelationEnlacement, vertic, horiz, numObjPixel)
    epsilonGlobalFG, epsilonLocalFG = spatialRelationEnlacementMeasure(matrixSpatialRelationEnlacementGlobalMethod, matrixSpatialRelationEnlacementLocalMethod, imageG )

    return epsilonGlobalFG, epsilonLocalFG
###################################################### Main code ######################################################

#Arguments in command line
ap = argparse.ArgumentParser()
ap.add_argument("-ri", "--referenceImage", required=True, help="Reference image")
ap.add_argument("-i", "--image", required=True, help="simple image")
args = vars(ap.parse_args())

#Read Images
imgF = cv2.imread(args["referenceImage"])
imgG =  cv2.imread(args["image"])

#Threshold & Binarize images
grayF = cv2.cvtColor(imgF, cv2.COLOR_BGR2GRAY) #Convert the reference Image to grayscale
grayG = cv2.cvtColor(imgG, cv2.COLOR_BGR2GRAY) #Convert the image to grayscale
ret, threshF = cv2.threshold(grayF, 127, 255, cv2.THRESH_BINARY)
ret, threshG = cv2.threshold(grayG, 127, 255, cv2.THRESH_BINARY)

#Resize image to same size
rF, cF = threshF.shape
rG, cG = threshG.shape
if(rF != rG or cF != cG): threshG = cv2.resize(threshG, (rF,cF))

#Enlacement FG
epsilonGlobalFG, epsilonLocalFG = calculateEnlacementMeasureFrom2Objects(threshF, threshG)

#Enlacement GF
epsilonGlobalGF, epsilonLocalGF = calculateEnlacementMeasureFrom2Objects(threshG, threshF)

#Interlacement
InterlacementFG_1, InterlacementFG_2 = calculateInterlacementMeasure(epsilonGlobalFG, epsilonLocalFG, epsilonGlobalGF, epsilonLocalGF)

print("EpsilonFG_1: {:.3f}, EpsilonFG_2: {:.3f}".format(epsilonGlobalFG, epsilonLocalFG))
print("EpsilonGF_1: {:.3f}, EpsilonGF_2: {:.3f}".format(epsilonGlobalGF, epsilonLocalGF))
print("InterlacementFG_1: {:.3f}, InterlacementFG_2: {:.3f}".format(InterlacementFG_1, InterlacementFG_2))
