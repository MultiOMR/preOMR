#!/home/alex/anaconda/bin/python

import sys, getopt
import preomr

def usage():
    print "Please provide input and output filenames."

def main(argv):
    if len(argv) < 3:
        usage()
        sys.exit(2)
    infile = argv[0]
    outfileA = argv[1]
    outfileB = argv[2]
    process(infile, outfileA, outfileB)

def process(infile, outfileA, outfileB):
    po = preomr.PreOMR(infile)
    # TODO make deskewing an option
    #po.deskew()
    #po.staffline_removal()
    if po.split_movements(outfileA, outfileB):
        print "success"
        sys.exit(0)
    else:
        print "no change of movement detected"
        sys.exit(1)

# Commandline parameter processing
# Make working folder
# Explode PDFs into PNGs?
# Deskew (optional)
# Find staves
# Print output image(s)

if __name__ == "__main__":
   main(sys.argv[1:])
