import sys
import cv2
import numpy as np
import math
import copy

from gamera.core import *
from gamera.toolkits.musicstaves import musicstaves_rl_roach_tatem
from gamera.toolkits.musicstaves import musicstaves_rl_fujinaga
from gamera.toolkits.musicstaves import stafffinder_miyao
from gamera.toolkits.musicstaves import stafffinder_projections
from gamera.plugins import numpy_io
init_gamera()

#import ossiafinder_dalitz

def intersect(r1,r2):
    """Returns the intersection of two rectangles"""
    x1 = max(r1['x'], r2['x'])
    y1 = max(r1['y'], r2['y'])
    x2 = min(r1['x'] + r1['width'],  r2['x'] + r2['width'])
    y2 = min(r1['y'] + r1['height'], r2['y'] + r2['height'])
    result = {"x": x1, "y": y1, "width": x2 - x1, "height": y2-y1}
    result['area'] = result['width'] * result['height']
    return(result)

def ydist(r1,r2):
    """distance on y-axis between two non-interecting rectangles"""
    top1 = r1['y']
    bottom1 = r1['y'] + r1['height']

    top2 = r2['y']
    bottom2 = r2['y'] + r2['height']
    return(min(abs(top1-bottom2), abs(top2-bottom1)))
    

def show(img, factor=0.5):
    """ show an image until the escape key is pressed
    :param factor: scale factor (default 0.5, half size)
    """
    if factor != 1.0:
        img = cv2.resize(img, (0,0), fx=factor, fy=factor) 

    cv2.imwrite('show.png',img)
#    while(1):
#        k = cv2.waitKey(0)
#        if k==27:    # Esc key to quit
#            cv2.destroyAllWindows()
#            exit()
#        if k==32:    # Space to stop
#            cv2.destroyAllWindows()
#            break


def deskew(img):
    """Deskews the given image based on lines detected with opencv's
    HoughLines function."""
    print "Deskewing."
    imgHeight, imgWidth, imgDepth = img.shape
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
                                              rotation,
                                              (imgWidth,imgHeight))
                           )
    return(deskewed)

class PreOMR(object):
    stavelineWidthThresh = 0.5
    
    def __init__(self, infile, deskew=False):
        self.debug = False
        self.infile = infile
        self.img = cv2.imread(self.infile)
        if deskew:
            self.img = deskew(self.img)
        self.original = self.img
        self.imgHeight, self.imgWidth, self.imgDepth = self.img.shape
        self.img_gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        if self.debug:
            self.debug_img = self.img.copy()
        ret2,self.img_binary = cv2.threshold(self.img_gray, 
                                             0,255,cv2.
                                             THRESH_BINARY+cv2.
                                             THRESH_OTSU)
    
    def staffline_removal(self):
        gamera_img = numpy_io.from_numpy(self.img)
        #self.save('tmp.png')
        #gamera_img = load_image('tmp.png')

        #ms = musicstaves_rl_roach_tatem.MusicStaves_rl_roach_tatem(gamera_img)
        ms = musicstaves_rl_fujinaga.MusicStaves_rl_fujinaga(gamera_img)
        cv2.imwrite('tmp.png', self.img)
        ms.remove_staves(crossing_symbols = 'bars')
        ms.image.save_PNG("tmpb.png")
        staffless = cv2.imread("tmp.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
        return(staffless)

    def find_staves(self, img):
        gamera_img = numpy_io.from_numpy(img)
        #sf = stafffinder_projections.StaffFinder_projections(gamera_img)
        #sf.find_staves(follow_wobble=True,preprocessing=0)
        sf = stafffinder_miyao.StaffFinder_miyao(gamera_img)
        sf.find_staves()
        #sf = ossiafinder_dalitz.Ossiafinder_dalitz(gamera_img)
        #sf.find_staves(debug=2)

        staves = sf.get_skeleton()
        #for i, staff in enumerate(staves):
        #    print "Staff %d has %d staves:" % (i+1, len(staff))
        #    for j, line in enumerate(staff):
        #        print("    %d. line at (%d,%d)" % (j+1,line.left_x,line.y_list[0]))
        return(staves)

    def find_blobs(self, img_binary):
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
            blob['boundingRect'] = rect
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

    def find_bars(self, system):
        staffless = self.staffline_removal()
        blobs = self.blobs

        """Finds the barlines in the system, given a binary image, a hash of
        info about the system, and blobs detected in the image.
        
        """
        img = system['image']

        for staff in system['staves']:
            min_x = 0
            max_x = self.imgWidth

            for line in staff:
                min_x = max(min_x, line.left_x)
                max_x = min(max_x, line.left_x + len(line.y_list))

                if self.debug:
                    for (i,y) in enumerate(line.y_list):
                        x = line.left_x + i
                        cv2.line(self.debug_img,(x,y),(x,y),(0,255,0),3)
                
#            cv2.line(img,(0,int(start)),(imgWidth,int(start)),(0,255,255),3)
#            cv2.line(img,(0,int(stop)),(imgWidth,int(stop)),(0,255,255),3)
#            cv2.line(img,(0,int(first_staveline)),(imgWidth,int(first_staveline)),(255,255,0),3)
#            cv2.line(img,(0,int(last_staveline)),(imgWidth,int(last_staveline)),(255,255,0),3)

            # assuming single staff for now..
            barlines = [0]
            system['barlines'] = barlines

            x_projection = []
            
            for x in range(min_x, max_x):
                first_staveline = staff[0].y_list[x - staff[0].left_x]
                last_staveline = staff[-1].y_list[x - staff[-1].left_x]

                #print("Stavelines: first %d last %d" % (first_staveline, last_staveline))
                stave_height = last_staveline - first_staveline

                # mean distance between stavelines
                avg_inter = float(stave_height) / float(len(staff)-1)
                #print("avg_inter: %f" % (avg_inter,))

                # where to look a bit above and below the stave for whitespace
                # above a barline
                gap = avg_inter / 2.0
                start = first_staveline - gap
                stop = last_staveline + gap

                # above stave, stave and below stave
                top = float(gap - 
                            cv2.countNonZero(staffless[start:first_staveline, 
                                                       x:x+1])) / float(gap)
                mid = float(stave_height - 
                            cv2.countNonZero(staffless[first_staveline:last_staveline,
                                                       x:x+1])
                           ) / float(stave_height)
                bot = float(gap - 
                            cv2.countNonZero(staffless[last_staveline:stop, x:x+1])
                           ) / float(gap)
                x_projection.append((top,mid,bot))
    
            barline_start = -1
            gap_dist = avg_inter/4
            gap_min = (avg_inter/float(stave_height)) * 0.3
            gap_tolerance = int(avg_inter/10)
            
            margin = int(avg_inter*2)
            
            for x in range(min_x+margin, max_x-margin):
                (top,mid,bot) = x_projection[x - min_x]
                #if self.debug:
                    #cv2.line(system['image'],(x,first_staveline),(x,int(first_staveline+((last_staveline-first_staveline)*mid))),(255,255,0),1)

                # found start of barline candidate
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
                            total = total + x_projection[i-min_x][1]
                        left_avg = total / ((gap_tolerance*2)+1)

                        total = 0
                        for i in range(right-gap_tolerance, right+gap_tolerance+1):
                            total = total + x_projection[i-min_x][1]
                        right_avg = total / ((gap_tolerance*2)+1)
                    
                        cv2.line(img,(left,first_staveline),(left,last_staveline),(255,0,255),1)
                        cv2.line(img,(right,first_staveline),(right,last_staveline),(255,0,255),1)

                        if (left_avg <= gap_min and right_avg <= gap_min):
                            #print("success: left_avg %f right_avg %f" % (left_avg, right_avg))
                            cv2.line(img,(barline_mid,first_staveline),(barline_mid,last_staveline),(255,0,0),3)
                            barlines.append(barline_mid)
                        else:
                            #print("fail: left_avg %f right_avg %f" % (left_avg, right_avg))
                            cv2.line(img,(barline_mid,first_staveline),(barline_mid,last_staveline),(0,255,0),3)
                        #show(img)
                        barline_start = -1
            (x1, y1, x2, y2) = system['location']
            #show(system['image'][y1:y2, x1:x2])

    def extract_bars(self, system, blobs):
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

            mask = np.zeros((self.imgHeight,self.imgWidth,1), np.uint8)

            cv2.drawContours(mask, contours, -1, 255, -1);
        
            inv = cv2.bitwise_not(img)
            dest = cv2.bitwise_and(inv,inv,mask = mask)
            dest = cv2.bitwise_not(dest)
            img_bar = dest[y1:y2, x1:x2]
            bar = {'image': img_bar,
                   'page': dest,
                   'location': [x1,y1,x2,y2]
            }
            result.append(bar)
            #show(img_bar)
        return(result)



    def find_staveblobs(self, cutstaves=False,img=None):
        if img == None:
            img = self.img
            img_binary = self.img_binary
        else:
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret2,img_binary = cv2.threshold(img_gray, 
                                            0,255,cv2.
                                            THRESH_BINARY+cv2.
                                            THRESH_OTSU)

        staves = self.find_staves(img)
        blobs = self.find_blobs(img_binary)

        staveblobs = []
        otherblobs = []

        for blob in blobs:
            rect = blob['rect']
            blob['staves'] = []
            blob['system'] = False
            if rect['width'] > (self.imgWidth * self.stavelineWidthThresh):
                for staff in staves:
                    inside = True
                    for staveline in staff:
                        leftmost = staveline.y_list[0]
                        # all stafflines have to be in blob
                        if leftmost < rect['y'] or leftmost > (rect['y'] + rect['height']):
                            inside = False
                            break
                    if inside:
                        blob['system'] = True
                        blob['staves'].append(staff)
            if blob['system']:
                staveblobs.append(blob)
                #print("found system with %d staves" % len(blob['staves']))
                if self.debug:
                    cv2.drawContours(self.debug_img,[blob['contour']],-1, (0, 0,255), 2)
            else:
                otherblobs.append(blob)
        return(staveblobs, otherblobs)

    def find_systems(self):
        img = self.img_binary
        #print "finding staves"
        (staveblobs, otherblobs) = self.find_staveblobs()
        #print("found %d staves" % (len(staveblobs),))
        blobs = staveblobs + otherblobs
        self.blobs = blobs
        # systems = []
        systems = staveblobs

        # attach disconnected bits in bounding box
        tidied = 0
        for blob in blobs:
            if not blob['system']:
                blob['parent'] = None
                for system in systems:
                    rect = intersect(system['rect'], blob['rect'])
                    if (rect['height'] > 0 and rect['width'] > 0):
                        # Biggest intersection wins
                        if (blob['parent'] == None) or (rect['area'] > blob['intersection']['area']):
                            blob['parent'] = system
                            blob['intersection'] = rect

                # Just assign to closest bounding rectangle on y-axis
                if blob['parent'] == None:
                    mindist = None
                    for system in systems:
                        dist = ydist(system['rect'], blob['rect'])
                        if mindist == None or mindist > dist:
                            blob['parent'] = system
                            mindist = dist
                    if blob['parent'] == None:
                        print "wtf"
                    else:
                        tidied = tidied + 1
        #print "tidied %d" % tidied

        # create new image for systems
        for system in systems:
            contours = [system['contour']]
            x1 = system['rect']['x']
            y1 = system['rect']['y']
            x2 = system['rect']['x'] + system['rect']['width']
            y2 = system['rect']['y'] + system['rect']['height']

            children = 0
            for blob in blobs:
                if blob['parent'] == system:
                    children = children + 1
                    contours.append(blob['contour'])
                    # include blob in image size/location
                    x1 = min(x1, blob['rect']['x'])
                    y1 = min(y1, blob['rect']['y'])
                    x2 = max(x2, blob['rect']['x'] + blob['rect']['width'])
                    y2 = max(y2, blob['rect']['y'] + blob['rect']['height'])

            #print("found %d children" % children)

            mask = np.zeros((self.imgHeight,self.imgWidth,1), np.uint8)

            cv2.drawContours(mask, contours, -1, 255, -1);
            #src = img[x1:y1, x2:y2]
            #srcMask = mask[y1:y2, x1:x2]
            kernel = np.ones((4,4),np.uint8)
            mask=cv2.dilate(mask,kernel,iterations=3)

            inv = cv2.bitwise_not(self.img)
            dest = cv2.bitwise_and(inv,inv,mask = mask)
            dest = cv2.bitwise_not(dest)

            (h,w,d) = dest.shape
            system['image'] = dest
            system['location'] = (x1, y1, x2, y2)
            system['height'] = h
            system['width'] = w

            min_x = self.imgWidth

            for staff in system['staves']:
                for line in staff:
                    min_x = min(min_x, line.left_x)

            system['stave_min_x'] = min_x

            #self.find_bars(system)
            #system['bar_images'] = self.extract_bars(system, blobs)
        if self.debug:
            cv2.imwrite('debug.png', self.debug_img)
        return(systems,blobs)

    def blob_image(self,img,blob):
        r = blob['rect']
        y1 = r['y']
        x1 = r['x']
        y2 = r['y'] + r['height']
        x2 = r['x'] + r['width']
        return(img[y1:y2, x1:x2])

    def remove_ossia(self):
        img = self.img
        
        ossia_mask = np.ones(self.img.shape[:2], dtype="uint8") * 255

        (staveblobs, otherblobs) = self.find_staveblobs()
        print("blobs %d/%d" % (len(staveblobs), len(otherblobs)))
        #staves = self.find_staves(img)

        working_img = img.copy()

        for blob in staveblobs:
            miny = self.imgHeight
            for staff in blob['staves']:
                staffline = staff[0]
                miny = min(min(staffline.y_list),miny)
            cv2.line(working_img, (0,miny-4), (self.imgWidth,miny-4), (255,255,255), 4) 

        cv2.imwrite('test.png', working_img)
        
        (staveblobs, otherblobs) = self.find_staveblobs(img=working_img)
        print("blobs %d/%d" % (len(staveblobs), len(otherblobs)))
        i = 0
        for blob in otherblobs:
            if blob['rect']['width'] < (self.imgWidth / 50):
                continue
            if blob['rect']['width'] > (self.imgWidth / 2):
                continue
            src = self.img
            mask = np.zeros((self.imgHeight,self.imgWidth,1), np.uint8)
            cv2.drawContours(mask, [blob['contour']], -1, (255,255,255), -1);
            inv = cv2.bitwise_not(src)
            dest = cv2.bitwise_and(inv,inv,mask = mask)
            dest = cv2.bitwise_not(dest)
            cropped = self.blob_image(dest, blob)

            gi = numpy_io.from_numpy(cropped)
            sf = stafffinder_projections.StaffFinder_projections(gi)
            sf.find_staves()
            staves = sf.get_skeleton()

            if len(staves) == 1 and len(staves[0]) >= 4:
                print("aha ossia with %d lines" % (len(staves[0]),))
                
                if self.debug:
                    cv2.imwrite('blobtest_%d.png' % i, dest)
                    i = i + 1
                cv2.drawContours(ossia_mask, [blob['contour']], -1, 0, -1)

        # erode a little to get rid of 'ghosting' around ossia
        kernel = np.ones((4,4),np.uint8)
        ossia_mask=cv2.erode(ossia_mask,kernel,iterations=4)
        
        cv2.imwrite('posterode.png', mask)
        
        result = img.copy()
        inverted = cv2.bitwise_not(result)
        result = cv2.bitwise_or(inverted,inverted,mask=ossia_mask)
        result = cv2.bitwise_not(result)

        self.img = result

    def split_movements(self, outfileA, outfileB):
        # 2% of page width
        indentThresh = 0.02 * self.imgWidth

        systems, blobs = self.find_systems()
        
        # Top - down order
        systems = sorted(systems, key=lambda system: system['rect']['y'])

        xs = []
        for system in systems:
            xs.append(system['stave_min_x'])
        threshold = min(xs) + indentThresh
        
        # Skip the first one, we don't split if the movement starts at top
        # of page
        found = None
        for i in range(1,len(systems)):
            #cv2.imwrite("system%d.png" %i, systems[i]['image'])
            if xs[i] > threshold:
                if found != None:
                    print "Oops, more than one movement found."
                found = i
                print("New movement at system %d" % (i+1))

        if (found):
            self.save_systems(outfileA, systems[:found])
            self.save_systems(outfileB, systems[found:])
        return(found)

    def save(self, outfile):
        cv2.imwrite(outfile, self.img)

    def save_systems(self, outfile, systems):
        print "saving %s" % outfile
        contours = []
        for system in systems:
            contours.append(system['contour'])

            for blob in self.blobs:
                if blob['parent'] == system:
                    contours.append(blob['contour'])

        mask = np.zeros((self.imgHeight,self.imgWidth,1), np.uint8)
        
        cv2.drawContours(mask, contours, -1, 255, -1);

        kernel = np.ones((4,4),np.uint8)
        mask=cv2.dilate(mask,kernel,iterations=1)
        
        inv = cv2.bitwise_not(self.img)
        dest = cv2.bitwise_and(inv,inv,mask = mask)
        dest = cv2.bitwise_not(dest)
        cv2.imwrite(outfile,dest)
