from __future__ import division, print_function
import functions as fn
import numpy as np
import scipy as sp


### Given n>=5 ground truths and their images in both cameras, compute the direct
### reconstruction that this corresponds to! 

## try also 10.4.3 : direct metric reconstruction using the IAC. Extract K from F,
## Then metric reconstruction of the scene may be computed by using the essential
## matrix as in section 9.6. Four solutions - only one physical, test with depth. 