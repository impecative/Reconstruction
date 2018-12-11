import numpy as np

loadfile = "imgpoints.npz"

outfile = np.load(loadfile)

imgpoints1 = outfile["coords1"]
imgpoints2 = outfile["coords2"]

print(imgpoints1.shape)