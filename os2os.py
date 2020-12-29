import sys
import os
import numpy as np
import cv2
import featureExtractor
import matplotlib.pyplot as plt
import math
import scipy.stats as st
from scipy.sparse import lil_matrix
from scipy import signal
from scipy import ndimage
from scipy.stats import norm
import DenseClusterFinder
import json
import time
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage.measurements import label
from skimage.measure import regionprops

randfiles = []
#plt = None
fig = None#plt.figure(figsize=(1,3))
pdf = st.norm.pdf
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.max()
    return kernel

def pdfsum(X,nsig,nsize):
    return st.norm.pdf(X/nsize,scale=nsig)/st.norm.pdf(0,scale=nsig)
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy
def centerOfMass(points,weights = None):
    if weights is None:
        return points.sum(axis=0)/points.shape[0]
    return (points.T * weights).sum(axis=1) / weights.sum()
def apply_kernel_to_points(coords,scores,imgSize,kernel):
    map = np.zeros((imgSize[0]+kernel.shape[0],imgSize[1]+kernel.shape[1]))
    kernelysize=kernel.shape[0]
    kernelxsize=kernel.shape[1]
    halfnsizex=int(kernelxsize/2)
    halfnsizey=int(kernelysize/2)
    i=0
    for voteCoord in coords:
        voteScore = scores[i]
        y1=voteCoord[1]+halfnsizey
        y2 = y1+kernelysize
        x1=voteCoord[0]+halfnsizex
        x2=x1+kernelxsize
        m=map[y1:y2,x1:x2]
        map[y1:y2,x1:x2] += kernel[:m.shape[0],:m.shape[1]]*voteScore
        i+=1
    return map[halfnsizex:-halfnsizex,halfnsizey:-halfnsizey]




    
def OS2OS_simple(img1,img1Rect,img2,depthFile=None,autoCenter=True, depth_quantization=1):
    if not autoCenter:
        roi = img1[img1Rect[1]:img1Rect[1]+img1Rect[3],img1Rect[0]:img1Rect[0]+img1Rect[2]]
    else:
        roi = img1
    
    
    
    t0 = time.time()
    img1FeatureStuct = featureExtractor.local_feature_detection_and_description("", 'SURF', 'SURF',
                                                                              5000, roi, mask=None,
                                                                              dense_descriptor=False,
                                                                              default_params=True)
    img2FeatureStuct = featureExtractor.local_feature_detection_and_description("", 'SURF', 'SURF',
                                                                                5000, img2, mask=None,
                                                                                dense_descriptor=False,
                                                                                default_params=True)
    t1 = time.time()
    img1Vectors = []
    f1 = img1FeatureStuct[1]
    f2  = img2FeatureStuct[1]
    # Get the roi center coordinates

    # subtract keypoint vectors from roi center so we can map matched keypoints from the database image to an "object center", divide out feature magnitude, subtract out the angle


    bf = cv2.BFMatcher()
#     print('feature sizes:',f1.shape,f2.shape)
    allMatches = bf.knnMatch(f1,f2,k=2)
    goodMatches = []
    goodMatchDists = []
    xpoints = []
    ypoints = []
    i=0
    matchToQ = {}
    useOneToOne=False

    for m, n in allMatches:
        if m.distance < .999 * n.distance:
            #goodMatches.append([m])
            #goodMatchDists.append(m.distance)
            qkp = img1FeatureStuct[0][m.queryIdx]
            dkp = img2FeatureStuct[0][m.trainIdx]
            if dkp in matchToQ:
                if matchToQ[dkp].distance > m.distance:
                    matchToQ[dkp] = m
            else:
                matchToQ[dkp] = m
            if not useOneToOne:
                goodMatchDists.append(m.distance)
                goodMatches.append([m])
                xpoints.append(qkp.pt[0])
                ypoints.append(qkp.pt[1])
        i+=1

    if useOneToOne:
        for dkp in matchToQ:
            m = matchToQ[dkp]
            goodMatches.append([m])
            goodMatchDists.append(m.distance)
            qkp = img1FeatureStuct[0][m.queryIdx]
            xpoints.append(qkp.pt[0])
            ypoints.append(qkp.pt[1])
#     print('good matches: ', len(goodMatches))
    #goodMatches = allMatches
    goodMatchDists = np.asarray(goodMatchDists)
    xpoints = np.array(xpoints)
    ypoints = np.array(ypoints)
    maxDist = goodMatchDists.max()
    minDist = goodMatchDists.min()
    normDists = (goodMatchDists-minDist)/(maxDist-minDist)
#     matchim2 = np.array([])
#     matchim2 = cv.drawMatches(img1, img1FeatureStuct[0], img2, img2FeatureStuct[0],
#                                [match[0] for match in goodMatches], matchim2)
#     cv.imwrite("./match.jpg",matchim2)
#     plt.figure(figsize=(30,30))
#     plt.imshow(matchim2)

    if autoCenter:
        goodMatchDistsInv = 1 / (1 + normDists)
        centerX = (xpoints*goodMatchDistsInv).sum()/goodMatchDistsInv.sum()
        centerY = (ypoints*goodMatchDistsInv).sum()/goodMatchDistsInv.sum()
        stdX = xpoints.std()
        stdY = ypoints.std()
        z=2
        roi1 = [centerX-(stdX*z)/2,centerY-stdY*z/2,stdX*z,stdY*z]
        g=np.bitwise_and(np.bitwise_and(xpoints >= roi1[0],xpoints < roi1[0]+roi1[2]),np.bitwise_and(ypoints >= roi1[1],ypoints < roi1[1]+roi1[2]))
        xpoints2 = xpoints[g]
        ypoints2 = ypoints[g]
        centerX2 = (xpoints2 * goodMatchDistsInv[g]).sum() / goodMatchDistsInv[g].sum()
        centerY2 = (ypoints2 * goodMatchDistsInv[g]).sum() / goodMatchDistsInv[g].sum()
        roiCenter = (centerX2,centerY2)

    for kp in img1FeatureStuct[0]:
        pt = kp.pt
        fangle = kp.angle*math.pi/180
        cvector=((roiCenter[0]-pt[0])/kp.size,(roiCenter[1]-pt[1])/kp.size)
        mag,ang = cart2pol(cvector[0],cvector[1])
        cvector = pol2cart(mag,ang-fangle)
        img1Vectors.append(cvector)

    matchScores = 1-normDists
    voteMap = np.zeros((img2.shape[0],img2.shape[1]),dtype=np.float32)
    nsize = int(math.pow(max(img2.shape[0],img2.shape[1])/4,.9))
    i = 0
    kernel = gkern(nsize,2)
    queryMeta = []
    databaseMeta = []
    scores = []
    goodscores = []
    count = 0

    voteMap = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.float32)

    i=0
    voteCoords = []
    originalCoords = []
    angles_g = []
    scales_g = []
    angles_p = []
    scales_p = []
    for m in goodMatches:
        scores.append(norm.pdf(m[0].distance))
#         scores.append(1/(1+m[0].distance))
    for m in goodMatches:
      
        im1KpIndex = m[0].queryIdx
        im2KpIndex = m[0].trainIdx
        kp = img2FeatureStuct[0][im2KpIndex]
        kp_probe = img1FeatureStuct[0][im1KpIndex]
        kpLoc = kp.pt
        kpAng = kp.angle*math.pi/180
        cvector = img1Vectors[im1KpIndex]
        mag,ang = cart2pol(cvector[0],cvector[1])
        cvector = pol2cart(mag,ang+kpAng)
        voteCoord = (int((kpLoc[0]+cvector[0]*kp.size)),int((kpLoc[1]+cvector[1]*kp.size)))
        nsize = kernel.shape[0]
        ks1 = int(nsize/2)
        ks2 = nsize-ks1
        if voteCoord[0] >= 0 and voteCoord[0] < img2.shape[1] and voteCoord[1] >= 0 and voteCoord[1] < img2.shape[0]:
            voteCoords.append(voteCoord)
            originalCoords.append(kpLoc)
#             voteMap[voteCoord[1]:voteCoord[1],voteCoord[0]:voteCoord[0]] += (kernel*scores[i])
            goodscores.append(scores[i])
            angles_g.append(kpAng)
            scales_g.append(kp.size)
            angles_p.append(kp_probe.angle)
            scales_p.append(kp_probe.size)
        i+=1
    angles_g = np.array(angles_g)
    scales_g = np.array(scales_g)
    angles_p = np.array(angles_p)
    scales_p = np.array(scales_p)
                        
    voteCoords = np.vstack(voteCoords)
    originalCoords = np.vstack(originalCoords)
#     img3 = cv.drawMatchesKnn(roi,img1FeatureStuct[0],img2,img2FeatureStuct[0],goodMatches,None)
    voteMin = voteMap.min()
    votemap = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.float32)
#     voteCoords += np.asarray([nsize/2,nsize/2],dtype=np.int)
    votemap[(voteCoords[:, 1], voteCoords[:, 0])] = goodscores
#     print('convolving...')
    votemap_conv = signal.fftconvolve(votemap,kernel,mode='same')
    
    votemap_blur= ndimage.gaussian_filter(votemap,20)
    
    maxs =  np.gradient(votemap_blur)
    grad = (maxs[0]+maxs[1])/2
#     print('maxs size: ', len(maxs))
#     maxs[maxs > 0] = 1
    
#     fig, axs = plt.subplots(6, 1, constrained_layout=True,figsize=(30,30))
#     axs[0].imshow(img2)
#     axs[0].set_title('gallery image')
#     axs[1].imshow(votemap_conv)
#     axs[1].set_title('votemap_conv')
    
#     axs[2].imshow(votemap_blur)
#     axs[2].set_title('votemap_blur')
#     axs[3].imshow(maxs[0])
#     axs[3].set_title('dif1')
#     axs[4].imshow(maxs[1])
#     axs[4].set_title('dif2')
#     axs[5].imshow(grad)
#     axs[5].set_title('grad')
    
    
    score = 1
    
    
    return votemap_blur,(centerX2,centerY2),score,voteCoords,originalCoords,angles_p,scales_p,angles_g,scales_g





def runOS2OSForFile(probeFile,imgFile,outfolder,depthFile=None,randomFiles=[]):
    if True:
        #probeFile = '/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/g121_dcmfmii.jpg'
        probeImg = cv2.imread(probeFile)
#         print('reading ',probeFile)

        rect = (0,0,probeImg.shape[0],probeImg.shape[1])
        folderPath = os.path.dirname(probeFile)
        allImageFiles = os.listdir(folderPath)
        centerColors = [(113,213,121),(247,0,255),(0,0,0),(76,190,251),(246,47,0),(17,12,162),(113,213,121),(255,255,255),(247,50,255)]
        allCenters = []
        imgNames= []
        for r in randomFiles:
            fileList.append(r)
        scoresForImages = []
        allimages = []
        times= []
        votemaps = []
#         for imgFile,depthFile in zip(fileList,depthFileList):
        #imgFile = os.path.join(folderPath,imgFile)
#             print('reading gallery',imgFile)
        if imgFile.endswith('.png') or imgFile.endswith('.jpg'):
            rImg = None
            try:
                rImg = cv2.imread(imgFile)
            except:
                print('cant read ', imgFile)
                return [None]*7
            if rImg is None or rImg.shape[0] <= 0 or rImg.shape[1] <= 0:
                print('image is none')
            else:

                ac = True
#                     print(rImg.shape)
                votemap,center,score,voteCoords,originalCoords,angles_p,scales_p,angles_g,scales_g = OS2OS_simple(probeImg,rect,rImg,depthFile,autoCenter=ac)
                votemaps.append(votemap)
                times.append(time)
                scoresForImages.append(score)
                allCenters.append(center)
                vmax = votemap.max()
                vmin = votemap.min()
                votemap_norm = np.power(((votemap-vmin)/(vmax-vmin)),1.5)
                #votemap_norm = ((votemap - vmin) / (vmax - vmin))
                if 'root' in imgFile:
                    print('root')
                outfile = os.path.join(outfolder,os.path.basename(imgFile))
                allimages.append(imgFile)
                cmap = plt.cm.jet
                norm = plt.Normalize(vmin=votemap_norm.min(), vmax=votemap_norm.max())
                cmapimg = (cmap(np.square(norm(votemap_norm)))[:,:,:-1]*255).astype(np.uint8)
                overlay = cmapimg.copy()
                output= rImg.copy()
                cv2.addWeighted(overlay, .5, rImg, .5, 0, output)
                imgNames.append(os.path.basename(imgFile))
                #plt.imsave(outfile+'.jpg', image)
                outpathfile = os.path.basename(outfile+'.jpg').split('_')[0]
                infolder = os.path.dirname(probeFile).split('/')[-1]
                outfile2 = os.path.join(outfolder,infolder)
#                     print(infolder,outfile2)
                try:
                    os.makedirs(outfile2)
                except:
                    pass
                outfile = os.path.join(outfile2,os.path.basename(outfile))
                outprobefile = os.path.join(outfile2,'query_'+os.path.basename(outfile))
#                     print(outfile)
#                     print(outprobefile)
                outprobe = probeImg.copy()

                outprobe = cv2.circle(outprobe,(int(center[0]),int(center[1])),int(max(outprobe.shape[0],outprobe.shape[1])/35),(255,0,0),-1)
#                     print('outfiles:' , outfile,outprobefile)

                cv2.imwrite(outfile+'.jpg',output)
                cv2.imwrite(outprobefile+'.jpg',outprobe)
                return votemaps,voteCoords,originalCoords,angles_p,scales_p,angles_g,scales_g




def runOS2OSForFileList(probeFile,fileList,outfolder,depthFileList=None,randomFiles=[]):
    if True:
        #probeFile = '/media/jbrogan4/scratch2/Reddit_Prov_Dataset_v5/Data/_Man_standing_on_clear_ice/g121_dcmfmii.jpg'
        probeImg = cv2.imread(probeFile)
#         print('reading ',probeFile)

        rect = (0,0,probeImg.shape[0],probeImg.shape[1])
        folderPath = os.path.dirname(probeFile)
        allImageFiles = os.listdir(folderPath)
        centerColors = [(113,213,121),(247,0,255),(0,0,0),(76,190,251),(246,47,0),(17,12,162),(113,213,121),(255,255,255),(247,50,255)]
        allCenters = []
        imgNames= []
        for r in randomFiles:
            fileList.append(r)
        scoresForImages = []
        allimages = []
        times= []
        votemaps = []
        for imgFile,depthFile in zip(fileList,depthFileList):
            #imgFile = os.path.join(folderPath,imgFile)
#             print('reading gallery',imgFile)
            if imgFile.endswith('.png') or imgFile.endswith('.jpg'):
                rImg = None
                try:
                    rImg = cv2.imread(imgFile)
                except:
                    print('cant read ', imgFile)
                    continue
                if rImg is None or rImg.shape[0] <= 0 or rImg.shape[1] <= 0:
                    print('image is none')
                else:
                   
                    ac = True
#                     print(rImg.shape)
                    votemap,center,score,voteCoords,originalCoords,angles_p,scales_p,angles_g,scales_g = OS2OS_simple(probeImg,rect,rImg,depthFile,autoCenter=ac)
                    votemaps.append(votemap)
                    times.append(time)
                    scoresForImages.append(score)
                    allCenters.append(center)
                    vmax = votemap.max()
                    vmin = votemap.min()
                    votemap_norm = np.power(((votemap-vmin)/(vmax-vmin)),1.5)
                    #votemap_norm = ((votemap - vmin) / (vmax - vmin))
                    if 'root' in imgFile:
                        print('root')
                    outfile = os.path.join(outfolder,os.path.basename(imgFile))
                    allimages.append(imgFile)
                    cmap = plt.cm.jet
                    norm = plt.Normalize(vmin=votemap_norm.min(), vmax=votemap_norm.max())
                    cmapimg = (cmap(np.square(norm(votemap_norm)))[:,:,:-1]*255).astype(np.uint8)
                    overlay = cmapimg.copy()
                    output= rImg.copy()
                    cv2.addWeighted(overlay, .5, rImg, .5, 0, output)
                    imgNames.append(os.path.basename(imgFile))
                    #plt.imsave(outfile+'.jpg', image)
                    outpathfile = os.path.basename(outfile+'.jpg').split('_')[0]
                    infolder = os.path.dirname(probeFile).split('/')[-1]
                    outfile2 = os.path.join(outfolder,infolder)
#                     print(infolder,outfile2)
                    try:
                        os.makedirs(outfile2)
                    except:
                        pass
                    outfile = os.path.join(outfile2,os.path.basename(outfile))
                    outprobefile = os.path.join(outfile2,'query_'+os.path.basename(outfile))
#                     print(outfile)
#                     print(outprobefile)
                    outprobe = probeImg.copy()

                    outprobe = cv2.circle(outprobe,(int(center[0]),int(center[1])),int(max(outprobe.shape[0],outprobe.shape[1])/35),(255,0,0),-1)
#                     print('outfiles:' , outfile,outprobefile)
                    
                    cv2.imwrite(outfile+'.jpg',output)
                    cv2.imwrite(outprobefile+'.jpg',outprobe)
                    return votemaps,voteCoords,originalCoords,angles_p,scales_p,angles_g,scales_g
#     except:
#         pass

    scoresForImages = np.array(scoresForImages)
    sortedinds = scoresForImages.argsort()[::-1]
    sortedscores = scoresForImages[sortedinds]
    filenames = np.array(allimages)[sortedinds]
    idwants=[]
    
    print('done')

def invsqr(x):
    return 1/(.000001+np.sqrt(x))

def drawObjects(image,object_locations):
    hulls = []
    for objs in object_locations:
        h = cv2.convexHull(objs.astype(np.int)).reshape(-1,2)
        hulls.append(h)
    vis = cv2.drawContours(image.copy(), hulls, -1, (0,255,0), 3)
    return vis

def os2osScoring(voteMap,voteCoords,originalCoords,angles_p,scales_p,angles_g,scales_g,depth_map=None):
    #First segment the voteMap
    local_thresh = threshold_local(voteMap, 31, offset=0)
    binary_local = voteMap > local_thresh
    img_cc, nb_cc = label(binary_local)
    
    #next, detect the blobs using adaptive thresholding
    cc = regionprops(img_cc,intensity_image=voteMap)
    object_candidates = np.array([c.max_intensity for c in cc])
    sorted_candidate_inds = np.argsort(object_candidates)
    
    #keep only the  blobs that have a max value contained in them in 95 percentile 
    mt = np.median(object_candidates)+object_candidates.std()
    good_inds = object_candidates >= mt
    
    
    #Loop through the vote blobs and re-score them according to 
    object_locations = []
    os2os_scores = []
    for ind in np.arange(len(good_inds))[good_inds]:
        blob = cc[ind]
        cords = np.fliplr(cv2.convexHull(blob.coords).reshape(-1,2))
        location_boundary = []
        angs = []
        sizes = []
        depths = []
        for i,vote_coord,original_coord in zip(range(voteCoords.shape[0]),voteCoords,originalCoords):
            inblob = cv2.pointPolygonTest(cords,tuple(vote_coord),False)
            if inblob >= 0:
                location_boundary.append(original_coord)
                angs.append(abs(angles_g[i]-angles_p[i]))
                sizes.append(scales_p[i]/scales_g[i])
                if depth_map is not None:
                    depths.append(depth_map[int(original_coord[0]),int(original_coord[1])])
        angs = np.asarray(angs)
        sizes = np.asarray(sizes)
        depths = np.asarray(depths)
        depth_score=1
        if depth_map is not None:
            #The more coherent all of the depths of the keypoints, the more likely it is one object
            depth_score=invsqr(depths.std())
        #Create a final os2os score
        os2os_score = object_candidates[ind]*invsqr(angs.std())*invsqr(sizes.std())*depth_score*100
        
        #filter out erroneous keypoint locations outside of the 98 percentile
        location_boundary = np.asarray(location_boundary)
        goodinds = (np.abs(location_boundary-location_boundary.mean(axis=0)) <= location_boundary.std(axis=0)*2).all(axis=1)
        location_boundary = location_boundary[goodinds]
        object_locations.append(location_boundary)
        os2os_scores.append(os2os_score)
    return object_locations,os2os_scores                                

