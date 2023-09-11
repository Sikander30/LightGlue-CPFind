# LightGlue-CPFind
CPFind replacement using LightGlue feature matching.

# Prerequisites:
Install LightGlue from: https://github.com/cvg/LightGlue \
A virtual environment is strongly recommended.

# Usage:
This script is intended to be a compatible replacement of CPFind (https://wiki.panotools.org/Cpfind) leveraging LightGlue to achieve better performance. \
Use the flag -h for usage information. Not all features of the original CPFind have been implemented yet.

# Warning:
This project is still in the early stages of development and has not been fully tested, usage in a production environment of any kind is not discouraged, use at your own risk. \
For still unclear reasons pictures aligned using the control points found by this program tend to be "curved" which results in reduced size in the crop of the final image, the problem is resolved by passing the "--straighten (-s)" option to pano_modify or by using the corresponding function in hugin GUI ("Move/Drag" tab -> "Straighten"). 
