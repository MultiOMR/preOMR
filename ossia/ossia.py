#!/home/alex/anaconda/bin/python

import sys, getopt
import preomr

def usage():
    print "Please provide input and output filenames."

def main(argv):
    if len(argv) < 2:
        usage()
        sys.exit(2)
    infile = argv[0]
    outfile = argv[1]

    process(infile, outfile)


def process(infile, outfile):
    po = preomr.PreOMR(infile)
    # TODO make deskewing an option
    po.deskew()
    #po.staffline_removal()
    po.find_ossia()
    po.save(outfile)

# Commandline parameter processing
# Make working folder
# Explode PDFs into PNGs?
# Deskew (optional)
# Find staves
# Print output image(s)

if __name__ == "__main__":
   main(sys.argv[1:])
