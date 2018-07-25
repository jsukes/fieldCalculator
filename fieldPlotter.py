import numpy as np
import matplotlib.pyplot as plt
import struct
import subprocess
import time

field_width = 20.0 	### physical width of the region being solved for (mm)
gridMajorDim = 64	### number of elements per side
def plotField():
	nx,ny,nz = gridMajorDim,gridMajorDim,gridMajorDim # dimensions of grid where signal intensity was solved for
	dx = field_width/nx	# width of each voxel
	
	### unpacks the binary data that the CUDA code generates
	struct_fmt = '{}{}{}'.format('= ',nx*ny*nz,'f')
	struct_len = struct.calcsize(struct_fmt)
	struct_unpack = struct.Struct(struct_fmt).unpack_from
	tmp = open('sig_field_bin',"rb").read(struct_len)
	c = np.array(struct_unpack(tmp),dtype=np.float)
	
	### populates a numpy array with the data loaded from the binary file
	field = np.zeros((nx,ny,nz))
	for z in range(0,nz):
		for y in range(0,ny):
			x0 = (z*gridMajorDim**2+y*gridMajorDim)
			field[z,y,:] = c[x0:(x0+gridMajorDim)]
			
	### opens a copy of GNUplot 
	pt = subprocess.Popen(['gnuplot','-e'],shell=True,stdin=subprocess.PIPE,)
	pt.stdin.write("set term x11 size 1500,1500 font 'Helvetica,80'; set pm3d map\n")
	### for whatever reason, the refresh rate of matplotlib is just awful. the solution is to pipe open gnuplot, write data files to disk, have GNUplot plot the contents of the data files
	
	### step through planes and plot the 2D signal intensity field at each level
	for z in range(0,nz):

		np.savetxt('fdata.dat',field[z,:,:],delimiter=' ')	# save file with the solved-for field
		time.sleep(100e-3) # wait to make sure the file has been written to disk
		
		### plot the result using gnuplot
		pt.stdin.write('{}{}{}{}{}{}{}{}{}{}{}'.format("set term x11 size 1500,1500 font 'Helvetica,80'; set pm3d map; set grid; set title 't=",z,"'; splot 'fdata.dat' u (($1-",nx/2,")*",dx,"):(($2-",ny/2,")*",dx,"):3 matrix with image noti\n"))
		#~ pt.stdin.write('{}{}{}{}{}{}{}{}{}{}{}'.format("set term x11 size 1500,1500 font 'Helvetica,80'; set pm3d map; set cbrange [-50:0]; set grid; set title 't=",z,"'; splot 'fdata.dat' u (($1-",nx/2,")*",dx,"):(($2-",ny/2,")*",dx,"):3 matrix with image noti\n"))
		
		time.sleep(0.05)


plotField()
