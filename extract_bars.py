#!/usr/bin/python

import sys
import segment
import pyPdf
import cv2

def get_pngs(source):
    (path, fn) = os.path.split(source)
    (name, ext) = os.path.splitext(fn)
    if ext.lower() == 'pdf':
        reader = pyPdf.PdfFileReader(open("foo.pdf"))
        pages = reader.getNumPages()



if len(sys.argv) > 1:
    source = sys.argv[1]
else:
    print("Please give a filename to convert")
    exit(-1);


systems = segment.segment(source)

for (i, system) in enumerate(systems):
    for (j, bar) in enumerate(system['bar_images']):
        fn = "bar_%03d_%03d.png" % (i, j)
        print "writing " + fn
        cv2.imwrite(fn, bar['image'])
