from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from delf import delf_config_pb2,feature_extractor


#import tensorflow as tf
import sys
import os

import json
import numpy as np
import time
import math
import subprocess
import traceback
from skimage.feature import local_binary_pattern
from extractAngles import getAnglesFromKeypoints
#from google.protobuf import text_format
#from tensorflow.python.platform import app

import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
hessianThreshold = 1000
nOctaves = 6
nOctaveLayers = 5
extended = False
upright = False
keepTopNCount = 2500
distanceThreshold = 50

_STATUS_CHECK_ITERATIONS = 100
config_path = '/home/jbrogan4/Documents/Projects/Medifor/tensorflow/models/research/delf/delf/python/examples/delf_config_example.pbtxt'
#config_path = '/scratch365/jbrogan4/models/research/delf/delf/python/examples/delf_config_example.pbtxt'
try:
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile(config_path, 'r') as f:
        text_format.Merge(f.read(), config)
except:
    print('no DELF')
def isRaw(imgname):
    return imgname.endswith('.raw') or imgname.endswith('.cr2') or imgname.endswith('.cr2') or imgname.endswith('.cr2') or imgname.endswith('.cr2') or imgname.endswith('.cr2') or imgname.endswith('.cr2') or imgname.endswith('.cr2')
def usage():
    print("extracts features")
def local_feature_detection(imgpath, img, detetype, kmax=500, mask=None, dense_descriptor=False, default_params=True):
    """ Sparsely detects local detection in an image.

    OpenCV implementation of various detectors.

    :param mask:
    :param imgpath:
    :param default_params:
    :param img: input image;
    :param detetype: type of detector {SURF, SIFT, ORB, BRISK}.
    :param kmax: maximum number of keypoints to return. The kmax keypoints with largest response are returned;

    :return: detected keypoins; detection time;
    """

    try:
        if detetype == "DELF":
            pass

        if detetype == "SURF":

            keypoints = []
            keypoints_surf = []
            keypoints_dense = []

            if default_params:
                surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000)
            else:
                #print("SURF: hessianThreshold = {0}".format(hessianThreshold))
                #print("SURF: nOctaves = {0}".format(nOctaves))
                #print("SURF: nOctaveLayers = {0}".format(nOctaveLayers))
                #print("Image Size: ",img.shape)
                sys.stdout.flush()

                surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)

            st_t = time.time()
            keypoints_surf = surf.detect(img, mask)

            ns = img.shape[:2]
            n_rows = ns[0]
            n_cols = ns[1]

            step_size = 5
            y_step_size = max(1,n_rows // 35)
            x_step_size = max(1,n_cols // 35)
            if (len(keypoints_surf) < (0.005*kmax)) or dense_descriptor:
                keypoints_dense = [cv2.KeyPoint(x, y, max(y_step_size, x_step_size)) for y in range(0, img.shape[0], y_step_size) for x in range(0, img.shape[1], x_step_size)]
                r_state = np.random.RandomState(7)
                keypoints_dense = list(r_state.permutation(keypoints_dense))

                print("Computing Dense Descriptor instead...", len(keypoints_dense))
                sys.stdout.flush()

                keypoints = keypoints_dense[0:kmax//2] + keypoints_surf

            else:
                keypoints = keypoints_surf

            ed_t = time.time()

            if kmax != -1:
                keypoints = keypoints[0:kmax]
        elif detetype == "SURF3":
            st_t = time.time()
            # detects the SURF keypoints, with very low Hessian threshold
            surfDetectorDescriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)
            keypoints = surfDetectorDescriptor.detect(img, mask)

            # sorts the keypoints according to their Hessian value
            keypoints = sorted(keypoints, key=lambda match: match.response, reverse=True)

            # obtains the positions of the keypoints within the described image
            positions = []
            for kp in keypoints:
                positions.append((kp.pt[0], kp.pt[1]))
            positions = np.array(positions).astype(np.float32)

            # selects the keypoints based on their positions and distances
            selectedKeypoints = []
            selectedPositions = []

            if len(keypoints) > 0:
                # keeps the top-n strongest keypoints
                for i in range(min(keepTopNCount,len(keypoints))):
                    selectedKeypoints.append(keypoints[i])
                    selectedPositions.append(positions[i])

                    # if the amount of wanted keypoints was reached, quits the loop
                    if len(selectedKeypoints) >= kmax:
                        break;

                selectedPositions = np.array(selectedPositions)

                # adds the remaining keypoints according to the distance threshold,
                # if the amount of wanted keypoints was not reached yet
                # print('selected keypoints size: ', len(selectedKeypoints), ' kmax: ',kmax)
                if len(selectedKeypoints) < kmax:
                    matcher = cv2.BFMatcher()
                    for i in range(keepTopNCount, positions.shape[0]):
                        currentPosition = [positions[i]]
                        currentPosition = np.array(currentPosition)

                        match = matcher.match(currentPosition, selectedPositions)[0]
                        if match.distance > distanceThreshold:
                            selectedKeypoints.append(keypoints[i])
                            selectedPositions = np.vstack((selectedPositions, currentPosition))

                        # if the amount of wanted keypoints was reached, quits the loop
                        if len(selectedKeypoints) >= kmax:
                            break;
                keypoints = selectedKeypoints
            ed_t = time.time()
        elif detetype == "SURF2":
            st_t = time.time()
            surfDetectorDescriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)
            keypoints = surfDetectorDescriptor.detect(img, mask)

            # sorts the keypoints according to their Hessian value
            keypoints = sorted(keypoints, key=lambda match: match.response, reverse=True)

            # obtains the positions of the keypoints within the described image
            positions = []
            for kp in keypoints:
                positions.append((kp.pt[0], kp.pt[1]))
            positions = np.array(positions).astype(np.float32)

            # selects the keypoints based on their positions and distances
            selectedKeypoints = []
            selectedPositions = []

            if len(keypoints) > 0:
                # keeps the top-n strongest keypoints
                for i in range(min(keepTopNCount,len(keypoints))):
                    selectedKeypoints.append(keypoints[i])
                    selectedPositions.append(positions[i])

                    # if the amount of wanted keypoints was reached, quits the loop
                    if len(selectedKeypoints) >= kmax:
                        break;

                selectedPositions = np.array(selectedPositions)

                # adds the remaining keypoints, avoiding collisions to the already selected ones
                if len(selectedKeypoints) < kmax:
                    matcher = cv2.BFMatcher()
                    for i in range(min(keepTopNCount,len(keypoints)), positions.shape[0]):
                        currentPosition = [positions[i]]
                        currentPosition = np.array(currentPosition)

                        match = matcher.match(currentPosition, selectedPositions)[0]
                        kp1 = selectedKeypoints[match.trainIdx]
                        kp2 = keypoints[i]

                        # collision detection
                        radiusSum = (kp1.size + kp2.size) / 2.0
                        distance = math.sqrt(
                            math.pow(kp1.pt[0] - kp2.pt[0], 2.0) + math.pow(kp1.pt[1] - kp2.pt[1], 2.0))
                        if distance > radiusSum:
                            selectedKeypoints.append(keypoints[i])
                            selectedPositions = np.vstack((selectedPositions, currentPosition))

                        # if the amount of wanted keypoints was reached, quits the loop
                        if len(selectedKeypoints) >= kmax or len(selectedKeypoints) == len(keypoints):
                            break;
                keypoints = selectedKeypoints
            ed_t = time.time()
        elif detetype == "SURF4":
            # detects the SURF keypoints
            surfDetectorDescriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)
            keypoints = surfDetectorDescriptor.detect(img, mask)

            # describes the obtained keypoints
            descriptions = []

            if len(keypoints) > 0:
                # removes the weakest keypoints (according to hessian)
                keypoints = sorted(keypoints, key=lambda match: match.response, reverse=True)
                if kmax != -1:
                    keypoints = keypoints[0:kmax]

        elif detetype == "KAZE":
            kaze = cv2.KAZE_create()
            st_t = time.time()
            keypoints = kaze.detect(img)
            ed_t = time.time()

        elif detetype == "SIFT":
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=kmax)
            st_t = time.time()
            keypoints = sift.detect(img)
            ed_t = time.time()

        elif detetype == "ORB":
            orb = cv2.ORB_create(nfeatures=kmax)
            st_t = time.time()
            keypoints = orb.detect(img)
            ed_t = time.time()

        elif detetype == "BRISK":
            brisk = cv2.BRISK_create()
            st_t = time.time()
            keypoints = brisk.detect(img)
            ed_t = time.time()

            keypoints = keypoints[0:kmax]

        elif detetype == "BINBOOST":
            current_path = os.path.dirname(__file__)
            binboost_exe = '{0}/boostDesc_1.0/./main'.format(current_path)
            matrices = '{0}/boostDesc_1.0/'.format(current_path)
            assert os.path.exists(binboost_exe), "BinBoost executable not found"

            cmd = '{0} --extract {1} {2}/.tmp.txt binboost {3}'.format(binboost_exe, imgpath, current_path, matrices)

            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

            output = []
            while True:
                line = p.stdout.readline()
                # print(line)
                if not line:
                    break
                output += [line.rstrip()]

            st_t, ed_t = np.array(output[0].split(), dtype=np.float32)

            keypoints = []
            for out in output[3:]:
                keypoints += [out.split()]

            if kmax != -1:
                keypoints = keypoints[0:kmax]

        elif detetype =="MSER_comp":
            ## This function takes two colored images as input and returns a similarity value
            def mserCompHist(img1, img2):
                mser = cv2.MSER()

                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                vis1 = img1.copy()
                vis2 = img2.copy()
                regions1 = mser.detect(gray1)
                hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions1]
                cv2.polylines(vis1, hulls, 1, (0, 255, 0))
                mask = np.zeros(gray1.shape, np.uint8)
                keypoints1 = []
                for hull in hulls:
                    (x, y), radius = cv2.minEnclosingCircle(hull)
                    center = (int(x), int(y))
                    radius = int(radius)
                    # cv2.circle(vis1, center, radius, (255, 0, 0), 2)
                    kp = cv2.KeyPoint()
                    kp.pt = center
                    kp.size = 2 * radius
                    keypoints1.append(kp)
                    cv2.drawContours(mask, [hull], 0, 255, -1)
                mask1 = cv2.bitwise_not(mask)
                masked_img1 = cv2.bitwise_and(gray1, gray1, mask=mask1)
                hist1 = cv2.calcHist([masked_img1], [0], None, [256], [0, 256])
                regions2 = mser.detect(gray2)
                hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions2]
                cv2.polylines(vis2, hulls, 1, (0, 255, 0))
                mask = np.zeros(gray2.shape, np.uint8)
                keypoints2 = []
                for hull in hulls:
                    (x, y), radius = cv2.minEnclosingCircle(hull)
                    center = (int(x), int(y))
                    radius = int(radius)
                    # cv2.circle(vis2, center, radius, (255, 0, 0), 2)
                    kp = cv2.KeyPoint()
                    kp.pt = center
                    kp.size = 2 * radius
                    keypoints2.append(kp)
                    cv2.drawContours(mask, [hull], 0, 255, -1)
                mask2 = cv2.bitwise_not(mask)
                masked_img2 = cv2.bitwise_and(gray2, gray2, mask=mask2)
                hist2 = cv2.calcHist([masked_img2], [0], None, [256], [0, 256])
                histSim = cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_INTERSECT)
                norm_histSim = histSim / (gray1.shape[0] * gray1.shape[1])
                return norm_histSim




        elif detetype == "MSER":

            st_t = time.time()

            mask = [[]]
            if img is None:
                return [], []

            # obtains the gray-scaled version of the given img
            gsImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # creates a mask to ignore eventual black borders
            _, bMask = cv2.threshold(cv2.normalize(gsImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX), 1, 255, cv2.THRESH_BINARY)
            bMask = cv2.convertScaleAbs(bMask)

            # combines the border mask to an eventual given mask
            if mask != [[]]:
                mask = cv2.bitwise_and(mask, bMask)
            else:
                mask = bMask

            # detects the keypoints
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gsImage)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

            keypoints = []
            for hull in hulls:
                (x, y), radius = cv2.minEnclosingCircle(hull)
                center = (int(x), int(y))
                radius = int(radius)

                kp = cv2.KeyPoint()
                kp.pt = center
                kp.size = 2 * radius
                keypoints.append(kp)

            print("-- MSER: NUMBER OF KEYPOINTS DETECTED:", len(keypoints))
            sys.stdout.flush()

            if len(keypoints) > kmax:
                print('-- MSER: SELECTING KEYPOINTS RANDOMLY!')
                sys.stdout.flush()

                r_state = np.random.RandomState(42)
                keypoints = list(r_state.permutation(keypoints))

            elif len(keypoints) == 0:
                print('-- MSER: DID NOT FOUND ANY KEYPOINT!')
                sys.stdout.flush()

                surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)

                keypoints = surf.detect(img, mask)

                if kmax != -1:
                    keypoints = keypoints[0:kmax]
            else:
                pass

            ed_t = time.time()

        elif detetype == "MSER_":

            st_t = time.time()

            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)

            keypoints_surf = surf.detect(img, mask)

            mask = [[]]

            # obtains the gray-scaled version of the given img
            gsImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # creates a mask to ignore eventual black borders
            _, bMask = cv2.threshold(cv2.normalize(gsImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX), 1, 255, cv2.THRESH_BINARY)
            bMask = cv2.convertScaleAbs(bMask)

            # combines the border mask to an eventual given mask
            if mask != [[]]:
                mask = cv2.bitwise_and(mask, bMask)
            else:
                mask = bMask

            # detects the keypoints
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gsImage)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

            keypoints_mser = []
            for hull in hulls:
                (x, y), radius = cv2.minEnclosingCircle(hull)
                center = (int(x), int(y))
                radius = int(radius)

                kp = cv2.KeyPoint()
                kp.pt = center
                kp.size = 2 * radius
                keypoints_mser.append(kp)

            r_state = np.random.RandomState(42)
            keypoints_mser = list(r_state.permutation(keypoints_mser))

            if len(keypoints_mser) < len(keypoints_surf):
                keypoints_mser = keypoints_mser[0:kmax//2]
                keypoints_surf = keypoints_surf[:kmax - len(keypoints_mser)]
            else:
                keypoints_surf = keypoints_surf[0:kmax//2]
                keypoints_mser = keypoints_mser[:kmax - len(keypoints_surf)]


            print('-- MSER: NUMBER OF KEYPOINTS', len(keypoints_mser))
            print('-- SURF: NUMBER OF KEYPOINTS', len(keypoints_surf))
            sys.stdout.flush()

            keypoints = keypoints_surf + keypoints_mser
            keypoints = keypoints[0:kmax]

            ed_t = time.time()

        else:
            ed_t, st_t = 0, 0
            keypoints = []

        det_t = ed_t - st_t
        return keypoints, det_t

    except:
        print("Failure in detecting the keypoints")
        sys.stdout.flush()
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return [], -1

def local_feature_description(img, keypoints, desctype, default_params=True,tfcores=1):
    """ Describes the given keypoints of an image.

    OpenCV implementation of various descriptors.

    :param default_params:
    :param img: input image;
    :param keypoints: computed keypoints;
    :param desctype: type of descriptor {SURF, SIFT, ORB, BRISK, RootSIFT}.

    :return: computed detection, description time.
    """
    newkeypoints = []
    if True:
        if desctype == "SURF" or desctype == "SURF2" or desctype == "SURF3":

            if default_params:
                surf = cv2.xfeatures2d.SURF_create()
            else:
                #print("SURF: hessianThreshold = {0}".format(hessianThreshold))
                #print("SURF: nOctaves = {0}".format(nOctaves))
                #print("SURF: nOctaveLayers = {0}".format(nOctaveLayers))
                sys.stdout.flush()

                surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)
                #print('got here 3')
            st_t = time.time()
            __, features = surf.compute(img, keypoints)
            ed_t = time.time()

        elif desctype == "SIFT":
            sift = cv2.xfeatures2d.SIFT_create()
            st_t = time.time()
            __, features = sift.compute(img, keypoints)
            ed_t = time.time()

        elif desctype == "LBP":

            radius = 5
            n_points = 64
            if len(img.shape) > 2 and img.shape[2] > 1:
                gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            else:
                gs = img
            features = np.zeros((len(keypoints),n_points))
            i = 0
            for kp in keypoints:
                neighborhoodSize = max(kp.size,30)
                top = max(0,kp.pt[1]-int(neighborhoodSize/2))
                bottom = min(img.shape[0],kp.pt[1]+int(neighborhoodSize/2))
                left = max(0,kp.pt[0]-int(neighborhoodSize/2))
                right = min(img.shape[1], kp.pt[0] + int(neighborhoodSize / 2))
                gspatch = gs[top:bottom,left:right]
                radius = max(2,min(radius,np.floor(gspatch.shape[0]/2),np.floor(gspatch.shape[1]/2)))
                lbp,dsc_t = local_binary_pattern(gspatch,n_points,radius,'uniform')
                h = np.histogram(lbp, normed=True, bins=n_points, range=(0, int(lbp.max() + 1)))
                h_norm = (h[0]*1.0)/np.sum(h[0])
                features[i,:] = h_norm
                i +=1

        elif desctype == "KAZE":
            # surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
            #                                    extended=extended, upright=upright)

            kaze = cv2.KAZE_create()

            st_t = time.time()
            __, features = kaze.compute(img, keypoints)
            ed_t = time.time()
        elif desctype == "DELF":

            from delf import feature_io
            # Extension of feature files.
            _DELF_EXT = '.delf'

            # Pace to report extraction log.
            _STATUS_CHECK_ITERATIONS = 100
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=tfcores,
                inter_op_parallelism_threads=tfcores,
                device_count={'GPU': 0})
            with tf.Graph().as_default():
                # Reading list of images.
                with tf.Session(config=session_conf) as sess:
                    # Initialize variables.
                    st_t = time.time()
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)

                    # Loading model that will be used.
                    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                               config.model_path)
                    graph = tf.get_default_graph()
                    input_image = graph.get_tensor_by_name('input_image:0')
                    input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
                    input_image_scales = graph.get_tensor_by_name('input_scales:0')
                    input_max_feature_num = graph.get_tensor_by_name(
                        'input_max_feature_num:0')
                    boxes = graph.get_tensor_by_name('boxes:0')
                    raw_descriptors = graph.get_tensor_by_name('features:0')
                    feature_scales = graph.get_tensor_by_name('scales:0')
                    attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
                    attention = tf.reshape(attention_with_extra_dim,
                                           [tf.shape(attention_with_extra_dim)[0]])

                    locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
                        boxes, raw_descriptors, config)

                    # Start input enqueue threads.
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    start = time.clock()
                    # Write to log-info once in a while.

                    # # Get next image.

                    # Extract and save features.
                    if len(img.shape) < 3:
                        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                    (newkeypoints, features, feature_scales_out,
                     attention_out) = sess.run(
                        [locations, descriptors, feature_scales, attention],
                        feed_dict={
                            input_image:
                                img[:, :, 0:3],
                            input_score_threshold:
                                config.delf_local_config.score_threshold,
                            input_image_scales:
                                list(config.image_scales),
                            input_max_feature_num:
                                config.delf_local_config.max_feature_num
                        })
                    kpList = []
                    for i in range(0,newkeypoints.shape[0]):
                        kpList.append(cv2.KeyPoint(newkeypoints[i][1],newkeypoints[i][0],feature_scales_out[i]))
                    try:
                        newkeypoints = getAnglesFromKeypoints(img,kpList,patchSize=150)
                    except:
                        print('could not extract angles')
                    # Finalize enqueue threads.
                    coord.request_stop()
                    coord.join(threads)
                    ed_t = time.time()
        elif desctype == "ORB":
            orb = cv2.ORB_create()
            st_t = time.time()
            __, features = orb.compute(img, keypoints)
            ed_t = time.time()

        elif desctype == "BRISK":
            brisk = cv2.BRISK_create()
            st_t = time.time()
            __, features = brisk.compute(img, keypoints)
            ed_t = time.time()

        elif desctype == "RootSIFT":
            eps = 0.00000001
            sift = cv2.xfeatures2d.SIFT_create()
            st_t = time.time()
            __, features = sift.compute(img, keypoints)

            features /= (np.sum(features, axis=1, keepdims=True) + eps)
            features = np.sqrt(features)

            ed_t = time.time()

        elif desctype == "BINBOOST":
            ed_t, st_t = 0, 0
            features = []
            for kp in keypoints:
                features += [kp[7:]]

            for i, kp in enumerate(keypoints):
                keypoints[i] = kp[:7]

            features = np.array(features, dtype=np.uint8)
            features = np.unpackbits(features, axis=1)

        elif desctype == "MSER":
            hessian = 100.0

            st_t = time.time()

            gsImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian)
            _, features = surf.compute(gsImage, keypoints)

            ed_t = time.time()

        else:
            ed_t, st_t = 0, 0
            features = []

        dsc_t = ed_t - st_t
        return features, dsc_t, 1,newkeypoints

    else:
        print("Failure in describing the keypoints")
        sys.stdout.flush()
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return [], -1, 0,[]

def local_feature_detection_and_description(imgpath, detetype, desctype, kmax=500, img=[], mask=None, dense_descriptor=False,
                                            default_params=True,tfcores=1):
    """ Given a path or an image, detects and describes local detection.

    :param default_params:
    :param mask:
    :param imgpath: path to the image
    :param detetype: type of detector {SURF, SIFT, ORB, BRISK}.
    :param desctype: type of descriptor {SURF, SIFT, ORB, BRISK, RootSIFT}.
    :param kmax: maximum number of keypoints to return. The kmax keypoints with largest response are returned;
    :param img: (optional) input image. If not present, loads the image from imgpath.

    :return: detected keypoints, described detection, detection time, description time.
    """
    if img == []:
        try:
                if imgpath.endswith('.gif'):
                    img = misc.imread(imgpath)
                else:
                    img = cv2.imread(imgpath)
                if img is None or img == []:
                    print("Could not open with OpenCV, trying raw codecs on ", imgpath)
                    img = rawpy.imread(imgpath).postprocess()
        except:
            print('Could Not load ', imgpath)
    keyps = None
    det_t=0
    if not detetype == 'DELF':
    # try:
        keyps, det_t = local_feature_detection(imgpath, img, detetype, kmax, mask, dense_descriptor, default_params)

        if not keyps:
            return None,None,None,None

    # misc.featuresize_lock.acquire()
    # misc.allImagefeaturesizes.append(len(keyps))
    # misc.featuresize_lock.release()

    feat, dsc_t, success, keyps2 = local_feature_description(img, keyps, desctype, default_params,tfcores)
    if keyps is None or len(keyps) == 0:
        keyps= keyps2
    #zeroimg = np.zeros(img.shape[:2])
    #for kp in keyps:
        #point = kp.pt
        #scale = kp.size
        #strength = kp.response
        #zeroimg = cv2.circle(zeroimg,(int(point[0]),int(point[1])),int(scale),int(strength),-1)

    if keyps == []: keyps = keyps2
    if feat is []:
        return keyps,[],None,None
    
    return keyps, feat, det_t, dsc_t

    # except ValueError:
    #     return [], [], -1, -1
def detect_and_describe(imgPaths,detetype,desctype,kmax,img,mask,dense_descriptor,default_params,basepath,newPath,tfcores=1):
    for im in imgPaths:
        relPath = os.path.relpath(im, basepath)
        newFullPath = os.path.join(newPath, 'features', relPath+'.npy')
        newDir = os.path.dirname(newFullPath)
        if not os.path.exists(newFullPath):
            f = local_feature_detection_and_description(im, detetype, desctype, kmax, [], mask, dense_descriptor,
                                                        default_params,tfcores)
            if f[0] is not None and f[1] is not None and len(f[0]) > 0 and len(f[1]) > 0:

                if not os.path.exists(newDir):
                    try:
                        os.makedirs(newDir)
                    except:
                        print('could not make path ', newDir)
                if os.path.exists(newDir):
                    np.save(newFullPath,f[1])
                    #print(im)
                    prog_q.put(im)
                else:
                    print('could not save file ', newFullPath)
            else:
                print('could not generate features for file ', im)
                unable_q.put(im)
        else:
            prog_q.put(im)
def progress_thread(fileList,newPath,machineNum,progjson):
    fileDict = {}
    completed = []
    unableList = []
    if progjson:
        completed = progjson['completedFiles']
        unableList = progjson['unableToCompleteFiles']

    pb = progressbar.ProgressBar(max_value=len(fileList))

    saveFileName = os.path.join(newPath,'extraction_progress','machine_'+str(machineNum)+'_prog.json')
    try:
        os.makedirs(os.path.dirname(saveFileName))
    except:
        pass
    for f in fileList:
        fileDict[f] = 1

    t0 = time.time()
    count = 0
    fcount = 0
    while count+fcount < len(fileList)-1:
        t1 = time.time()
        f = prog_q.get()
        if f in fileDict:
            del fileDict[f]
            count +=1
        completed.append(f)
        pb.update(count)
        if unable_q.qsize() > 0:
            unableList.append(unable_q.get())
            fcount +=1
        if t1-t0 > 120:
            remainingFiles = sorted(list(fileDict.keys()))
            d = {}
            d['uncompletedFiles'] = remainingFiles
            d['unableToCompleteFiles'] = unableList
            d['completedFiles'] = completed
            print('saving progress...')
            with open(saveFileName,'w')as fp:
                json.dump(d,fp)
            print('progress saved!')
            t0=time.time()

    print('progress thread quit on call', count+fcount, len(fileList))

def recalcProgressWithoutFiles(newpath,jsonpath,featureDirectory):
    featureFile_dirs = os.listdir(featureDirectory)
    for d in featureFile_dirs:
        bar=progressbar.ProgressBar()
        featureFiles = os.listdir(os.path.join(featureDirectory,d))
 
def recalcProgress(newPath,jsonpath):
    progfilepath = os.path.join(newPath, 'extraction_progress')
    if os.path.exists(progfilepath):
        newLeft = []
        newCouldnt = []
        print('looking in ', progfilepath)
        for f in os.listdir(progfilepath):
            if f.endswith('.json'):
                print('found progress file ', f)
                with open(os.path.join(progfilepath,f),'r') as fp:
                    j = json.load(fp)
                newLeft += j['uncompletedFiles']
                newCouldnt += j['unableToCompleteFiles']
        with open (jsonpath,'r') as fp:
            fulljson = json.load(fp)
        fulljson['uncompletedFiles']=newLeft
        fulljson['unableToCompleteFiles']=newCouldnt
        with open(jsonpath,'w') as fp:
            json.dump(fulljson,fp)
        print('saved new json file to ', jsonpath)
#     Set up threading queues for progress calculation

if __name__ == "__main__":
    import progressbar
    import rawpy
    from joblib import Parallel, delayed, load, dump
    from multiprocessing import Process
    from multiprocessing import Manager
    from scipy import misc

    stillRunning = Manager().Value('j', True)
    prog_q = Manager().Queue(100000)
    unable_q = Manager().Queue(1000)

    args = sys.argv[1:]
    jsonFile = None
    numCores = 1
    numJobs = 1
    machineNum = 0
    threadBatch = 1
    kmax = 1000
    outputDir = None
    detType = 'SURF'
    descType = 'SURF'
    datasetName = ''
    recalcProg = False
    index_key = None
    machineOffset = 0
    features = detect_and_describe(['/home/jbrogan4/Documents/Projects/Medifor/tensorflow/models/research/delf/delf/python/examples/snowwhite.jpg'],'SURF','SURF',5000,[],None,False,True,'','./')
    while args:
        a = args.pop(0)
        if a == '-h':
            usage()
            sys.exit(1)
        elif a == '-jsonFile':      jsonFile  = args.pop(0)
        elif a == '-numCores':      numCores = int(args.pop(0))
        elif a == '-numJobs':       numJobs = int(args.pop(0))
        elif a == '-machineNum':    machineNum = int(args.pop(0))-1
        elif a == '-threadBatch':   threadBatch = int(args.pop(0))
        elif a == '-detectType':    detType = args.pop(0)
        elif a == '-descType':      descType = args.pop(0)
        elif a == '-kmax':          kmax = int(args.pop(0))
        elif a == '-outputDir' :     outputDir = args.pop(0)
        elif a == '-datasetName' :  datasetName = args.pop(0)
        elif a == '-recalcProgress' : recalcProg = True
        elif a == '-machineOffset' : machineOffset = int(args.pop(0))
        elif not index_key:
            index_key = a
        else:
            print("argument %s unknown" % a)
            sys.exit(1)
    machineNum -= machineOffset
    outputDir = os.path.join(outputDir, datasetName, descType+'_'+detType)
    if recalcProg:
        recalcProgress(outputDir,jsonFile)
    with open(jsonFile,'r') as f:
        indexJson = json.load(f)

    progFile = os.path.join(outputDir,'extraction_progress','machine_'+str(machineNum)+'_prog.json')
    progJson = None
    if os.path.exists(progFile):
        with open(progFile,'r') as fp:
            progJson = json.load(fp)
    if 'uncompletedFiles' in indexJson:
        print('index json contains uncompleted files to run: ',len(indexJson['uncompletedFiles']))
        fileList = indexJson['uncompletedFiles']
    elif progJson and 'uncompletedfiles' in progJson:
        fileList = progJson['uncompletedFiles']
        print('Found progress file, ',len(fileList),' of ',len(indexJson['imageList']),' files left to process')
    else:
        fileList = indexJson['imageList']
    fileList = sorted(fileList)
    baseDir = indexJson['baseDir']
    machinePartitionSize = int(float(len(fileList))/float(numJobs))
    filePart = fileList[machinePartitionSize*machineNum:min(len(fileList), machineNum*machinePartitionSize+machinePartitionSize)]
    print('total number of files: ',len(fileList))
    print('files to process in this job: ',len(filePart))
    batches = []
    count = 0
    p0 = Process(target=progress_thread, args=(filePart,outputDir,machineNum,progJson), )
    p0.start()
    while count < len(filePart):
        b = filePart[count:min(len(filePart),count+threadBatch)]
        count+=threadBatch
        batches.append(b)
    print('number of batches: ', len(batches))
    if numJobs == 1:
        for b in batches:
            detect_and_describe(b,detType,descType,kmax,[],None,False,True,baseDir,outputDir)
    else:
        counts = Parallel(n_jobs=numJobs)(delayed(detect_and_describe)(b,detType,descType,kmax,[],None,False,True,baseDir,outputDir) for b in batches)
    stillRunning.value = False
    p0.join()
