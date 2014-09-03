import sys
import cv2
import numpy as np
import math
import copy

# threshold at 50%
stavelineWidthThresh = 0.5

# used for visualising progress..
output = None

def show(img, factor=0.5):
    """ show an image until the escape key is pressed
    :param factor: scale factor (default 0.5, half size)
    """
    if factor != 1.0:
        img = cv2.resize(img, (0,0), fx=factor, fy=factor) 

    cv2.imshow('image',img)
    while(1):
        k = cv2.waitKey(0)
        if k==27:    # Esc key to stop
            break
    cv2.destroyAllWindows()

def deskew(img):
    """Deskews the given image based on lines detected with opencv's
    HoughLines function."""
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray,50,150,apertureSize = 3)
    minLineLength = int(imgWidth*0.5)
    houghThresh = int(imgWidth*0.15)
    maxLineGap = 10
    #lines = cv2.HoughLinesP(img_edges,1,np.pi/(180*1),houghThresh,minLineLength,maxLineGap)
    lines = cv2.HoughLines(img_edges,1,np.pi/(180*3),houghThresh)

    angles = []
    for rho,theta in lines[0]:
        
        angles.append((theta - (np.pi / 2)))

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + imgWidth*(-b))
        y1 = int(y0 + imgWidth*(a))
        x2 = int(x0 - imgWidth*(-b))
        y2 = int(y0 - imgWidth*(a))
        #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

    middle = np.median(angles)
    middle_deg = middle * (180/np.pi)

    rotation = cv2.getRotationMatrix2D((imgWidth/2,imgHeight/2),middle_deg,1.0)

    # rotate while inverted. the background is filled with zeros
    # (black), this inversion means that ends up white
    deskewed = cv2.bitwise_not(cv2.warpAffine(cv2.bitwise_not(img),
                                              rotation,(imgWidth,imgHeight)))
    return(deskewed)

def find_blobs(img_binary):
    """Find blobs in the given image, returned as a list of associative
    lists containing various cheap metrics for each blob."""

    blobs = []
    img_inverted = cv2.bitwise_not(img_binary)
    contours, hierarchy = cv2.findContours(img_inverted,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for (i, c) in enumerate(contours):
        blob = {}
        blobs.append(blob)
        
        blob['area'] = cv2.contourArea(c)
        
        m = cv2.moments(c)
        if m['m00'] == 0: # When would this be true?
            blob['x'] = 0
            blob['y'] = 0
        else:
            blob['x'] = m['m10'] / m['m00']
            blob['y'] = m['m01'] / m['m00']
      
        blob['contour'] = c

        rect = cv2.boundingRect(c)
        blob['rect'] = {'x': rect[0], 
                        'y': rect[1], 
                        'width': rect[2], 
                        'height': rect[3]
                       }
        blob['hull'] = hull = cv2.convexHull(c)
        blob['hull_area'] = abs(cv2.contourArea(hull))
        blob['system'] = False
        blob['parent'] = None
        #blob['perimeter'] = perimeter = cv2.arcLength(c, True)
        #blob['roundness'] = (perimeter * 0.282) / math.sqrt(area)
        #(centre, axes, orientation) = cv2.fitEllipse(c)
        #blob['orientation'] = orientation / 180
        #print "orientation: %f" % orientation
        #blob['aspect'] = float(rect[1]) / float(rect[3])

    return blobs


def find_stavelines(img):
    """Finds potential stavelines, using threshold of y-projection."""
    result = []
    y_projection = []
    total = 0

    for y in range(0, imgHeight):
        # count number of black pixels in row. This count of a
        # numpy-style range is *much* faster than counting pixel by
        # pixel.
        b = imgWidth - cv2.countNonZero(img[y:y+1, 0:imgWidth])
        y_projection.append(b)
        total = total + b

    lines = []

    line_start = -1
    for y in range(0, imgHeight):
        b = y_projection[y]
        #print("b: %d thresh: %d" % (b, imgWidth * stavelineWidthThresh))
        if b > (float(imgWidth) * stavelineWidthThresh):
            if line_start < 0:
                line_start = y
        else:
            if line_start >= 0:
                middle = (line_start + (y-1)) / 2
                lines.append(middle)
                if output:
                    cv2.line(output,(0,middle),(imgWidth,middle),(255,0,255),3)
                    cv2.line(output,(0,middle),(b,middle),(255,0,0),2)
                line_start = -1
    return(lines)

def intersect(r1,r2):
    """Returns the intersection of two rectangles"""
    x1 = max(r1['x'], r2['x'])
    y1 = max(r1['y'], r2['y'])
    x2 = min(r1['x'] + r1['width'],  r2['x'] + r2['width'])
    y2 = min(r1['y'] + r1['height'], r2['y'] + r2['height'])
    result = {"x": x1, "y": y1, "width": x2 - x1, "height": y2-y1}
    result['area'] = result['width'] * result['height']
    return(result)

def extract_bars(system, blobs):
    """Given information about a system (including identified barlines),
    and all the blobs on a page returns a list of bars in the system,
    each an associative array containing image and location.

    """

    img = system['image']

    barlines = system['barlines']

    result = []

    for i in range(0,len(barlines)):
        barstart = barlines[i]
        if i == (len(barlines)-1):
            barstop = system['width']
        else:
            barstop = barlines[i+1]
        #print("barstart %d barstop %d" % (barstart, barstop))
        contours = [system['contour']]
        x1 = barstart
        y1 = system['location'][1]
        x2 = barstop
        y2 = system['location'][3]
        h = y2 - y1
        w = x2 - x1
        #print("height %d width %d" % (h, w))
        for blob in blobs:
            if blob['parent'] == system:
                if blob['rect']['x'] >= barstart and blob['rect']['x'] + blob['rect']['width'] <= barstop:
                    contours.append(blob['contour'])

        mask = np.zeros((imgHeight,imgWidth,1), np.uint8)

        cv2.drawContours(mask, contours, -1, 255, -1);

        inv = cv2.bitwise_not(img)
        dest = cv2.bitwise_and(inv,inv,mask = mask)
        dest = cv2.bitwise_not(dest)
        img_bar = dest[y1:y2, x1:x2]
        bar = {'image': img_bar,
               'location': [x1,y1,x2,y2]
        }
        result.append(bar)
        #show(img_bar)
    return(result)

def find_bars(img_binary, system, blobs):
    """Finds the barlines in the system, given a binary image, a hash of
    info about the system, and blobs detected in the image.

    """
    img = system['image']
    first_staveline = system['stavelines'][0]
    last_staveline = system['stavelines'][-1]
    stave_height = last_staveline - first_staveline
    # mean distance between stavelines
    avg_inter = float(last_staveline - first_staveline) / float(len(system['stavelines'] )-1)
    gap = avg_inter / 2.0

    # where to look a bit above and below the stave for whitespace
    # above a barline
    start = first_staveline - gap
    stop = last_staveline + gap

    #cv2.line(img,(0,first_staveline),(w,first_staveline),(0,255,255),3)
    #cv2.line(img,(0,last_staveline),(w,last_staveline),(0,255,255),3)

    barlines = [0]
    system['barlines'] = barlines
    x_projection = []
    for x in range(0, system['width']):
        # above stave, stave and below stave
        top = float(gap - 
                    cv2.countNonZero(img_binary[start:first_staveline, 
                                                x:x+1])) / float(gap)
        mid = float(stave_height - 
                    cv2.countNonZero(img_binary[first_staveline:last_staveline,
                                                x:x+1])
                   ) / float(stave_height)
        bot = float(gap - 
                    cv2.countNonZero(img_binary[last_staveline:stop, x:x+1])
                   ) / float(gap)
        x_projection.append((top,mid,bot))
    
    barline_start = -1
    gap_dist = avg_inter/4
    gap_min = (avg_inter/float(stave_height)) * 1.3
    gap_tolerance = int(avg_inter/10)

    for x in range(0, system['width']):
        (top,mid,bot) = x_projection[x]
        #print("mid: %f top %f bot: %f" % (mid,top,bot))
        if output:
            cv2.line(system['image'],(x,first_staveline),(x,int(first_staveline+((last_staveline-first_staveline)*mid))),(255,255,0),1)

        if top < 0.6 and bot < 0.6 and mid > 0.95:
            if barline_start < 0:
                barline_start = x
        else:
            if barline_start > 0: 
                # check there is nothing either side of 'barline'
                barline_stop = x-1
                barline_mid = barline_stop - ((barline_stop - barline_start)/2)
                #print("barline start %d stop %d mid %d" % (barline_start, barline_stop, barline_mid))
                left = int(max(0,barline_start-gap_dist))
                right = int(min(system['width']-1,(x-1)+gap_dist))

                total = 0
                for i in range(left-gap_tolerance, left+gap_tolerance+1):
                    total = total + x_projection[i][1]
                left_avg = total / ((gap_tolerance*2)+1)

                total = 0
                for i in range(right-gap_tolerance, right+gap_tolerance+1):
                    total = total + x_projection[i][1]
                right_avg = total / ((gap_tolerance*2)+1)
                    
                if output:
                    cv2.line(system['image'],(left,first_staveline),(left,last_staveline),(255,0,255),1)
                    cv2.line(system['image'],(right,first_staveline),(right,last_staveline),(255,0,255),1)

                if (left_avg <= gap_min and right_avg <= gap_min):
                    barlines.append(barline_mid)
                    if output:
                        cv2.line(system['image'],(barline_mid,first_staveline),(barline_mid,last_staveline),(255,0,0),3)
                else:
                    if output:
                        cv2.line(system['image'],(barline_mid,first_staveline),(barline_mid,last_staveline),(255,0,0),3)
                        cv2.line(system['image'],(barline_mid,first_staveline),(barline_mid,last_staveline),(0,255,0),3)
                barline_start = -1
    (x1, y1, x2, y2) = system['location']
    #show(system['image'][y1:y2, x1:x2])

def preprocess(img):
    global imgHeight, imgWidth, imgDepth

    imgHeight, imgWidth, imgDepth = img.shape
    img = deskew(img)

    # Binarisation
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret2,img_binary = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img, img_binary


def find_systems(img, img_binary):
    stavelines = find_stavelines(img_binary)
    blobs = find_blobs(img_binary)
    systems = []

    for blob in blobs:
        rect = blob['rect']
        if rect['width'] > (imgWidth * stavelineWidthThresh):
            staveline_count = 0
            found_stavelines = []
            for staveline in stavelines:
                if staveline >= rect['y'] and staveline <= (rect['y'] + rect['height']):
                    found_stavelines.append(staveline)
            if len(found_stavelines) >= 5:
                blob['system'] = True
                blob['stavelines'] = found_stavelines
                systems.append(blob)
                #print("found system with %d stavelines" % len(found_stavelines))
                if output:
                    cv2.drawContours(output,[blob['contour']],-1, (0, 0,255), 2)
            else:
                #print("didn't find system with %d stavelines" % staveline_count)
                pass

    # attach disconnected bits in bounding box
    for blob in blobs:
        if not blob['system']:
            for system in systems:
                rect = intersect(system['rect'], blob['rect'])
                if (rect['height'] > 0 and rect['width'] > 0):
                    isParent = False
                    if blob['parent'] == None:
                        isParent = True
                    else:
                        # Biggest intersection wins
                        if rect['area'] > blob['intersection']['area']:
                            isParent = True
                    if isParent:
                        blob['parent'] = system
                        blob['intersection'] = rect
                        if output:
                            cv2.drawContours(output,[blob['contour']],-1, (0, 255,0), 2)

    # create new image for systems
    for system in systems:
        contours = [system['contour']]
        x1 = system['rect']['x']
        y1 = system['rect']['y']
        x2 = system['rect']['x'] + system['rect']['width']
        y2 = system['rect']['y'] + system['rect']['height']

        for blob in blobs:
            if blob['parent'] == system:
                contours.append(blob['contour'])
                # include blob in image size/location
                x1 = min(x1, blob['rect']['x'])
                y1 = min(y1, blob['rect']['y'])
                x2 = max(x2, blob['rect']['x'] + blob['rect']['width'])
                y2 = max(y2, blob['rect']['y'] + blob['rect']['height'])

        mask = np.zeros((imgHeight,imgWidth,1), np.uint8)

        cv2.drawContours(mask, contours, -1, 255, -1);
        #src = img[x1:y1, x2:y2]
        #srcMask = mask[y1:y2, x1:x2]
        inv = cv2.bitwise_not(img)
        dest = cv2.bitwise_and(inv,inv,mask = mask)
        dest = cv2.bitwise_not(dest)

        (h,w,d) = dest.shape
        system['image'] = dest
        system['location'] = (x1, y1, x2, y2)
        system['height'] = h
        system['width'] = w
        find_bars(img_binary, system, blobs)
        system['bar_images'] = extract_bars(system, blobs)
    return(systems, blobs)

def segment(source):

    img = cv2.imread(source)

    # Noise reduction
    # denoised = cv2.fastNlMeansDenoising(img)

    img, img_binary = preprocess(img)

    systems, blobs = find_systems(img, img_binary)
    return(systems)
