import os
import sys
import warnings
import jax.numpy as jnp
from jax.config import config
from jax import tree_util
import jax
import jax.profiler
from enum import Enum
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import time
import MemPrO as ori
import argparse
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils
from scipy.optimize import minimize



#We define some enumerations for use later
class Reses(Enum):
	ALA = 0
	GLY = 1
	ILE = 2
	LEU = 3
	PRO = 4
	VAL = 5
	PHE = 6
	TYR = 7
	TRP = 8
	ARG = 9
	LYS = 10
	HIS = 11
	SER = 12
	THR = 13
	ASP = 14
	GLU = 15
	ASN = 16
	GLN = 17
	CYS = 18
	MET = 19
	
class Beads(Enum):
	BB =  0
	SC1 = 1
	SC2 = 2
	SC3 = 3
	SC4 = 4
	SC5 = 5
	SC6 = 6
	
class Beadtype(Enum):
	SP2 = 0 #p4
	TC3 = 1 #p4
	SP1 = 2 #P1
	P2 = 3 #p5
	SC2 = 4 #AC2
	SP2a = 5 #p5
	SC3 = 6 #c3
	SC4 = 7 #SC5
	TC5 = 8 #SC5
	TC4 = 9 #SC4
	TN6 = 10 #SP1
	TN6d = 11 #SNd
	SQ3p = 12 #Qp
	SQ4p = 13 #Qp
	TN5a = 14 #SP1
	TP1 = 15 #P1
	SQ5n = 16 #Qa
	Q5n = 17 #Qa
	TC6 = 18 #c5
	C6 = 19 #c5
	P5 = 20 #p5
	SP5 = 21 #p4
	GHOST = 22
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
config.update("jax_enable_x64",True)

martini_default_path = os.environ["PATH_TO_MARTINI"]

#Reading command line args
parser = argparse.ArgumentParser(prog="MemPrOD",description="This is a program for the prediction of deformations caused by membrane proteins.")
parser.add_argument("-f", "--file_name",help = "Input file name (.pdb)")
parser.add_argument("-o","--output",help="Name of the output directory (Default: Deformations)")
parser.add_argument("-ni","--iters",help="Number of minimisation iterations (Default: 75)")
parser.set_defaults(iters=75)
parser.add_argument("-ng","--grid_density",help="Spacing of grid points (Default: 3)")
parser.set_defaults(grid_density=3)
parser.add_argument("-ncav","--no_cavity_surface",action="store_false",help="Toggle use of a surface finder that is more accurate but more expensive.")
parser.set_defaults(no_cavity_surface=True)
parser.add_argument("-itp","--itp_file",help="Path to force field (martini_v3.itp)")
parser.set_defaults(itp_file=martini_default_path)
parser.add_argument("-bd","--build_system",help = "Build a MD ready CG-system for ranks < n (Default: n=0)")
parser.set_defaults(build_system=False)
parser.add_argument("-bd_args","--build_arguments",help="Arguments to pass to insane when building system")
parser.set_defaults(build_arguments="")
parser.add_argument("-cut","--cutoff",help="Cutoff for LJ interactions (Deafult: 12)")
parser.set_defaults(cutoff=12)
parser.add_argument("-k1","--k1_const",help="Spring constant for neighbouring grid point Z deviations (Deafult: 0.25)")
parser.set_defaults(k1_const=0.25)
parser.add_argument("-k2","--k2_const",help="Spring constant for membrane compression (Deafult: 0)")
parser.set_defaults(k2_const=0)
parser.add_argument("-zshift","--zshift",help="Z shift of membrane for proteins that are missoriented due to lack of deformations.")
parser.set_defaults(zshift=0)
parser.add_argument("-mt","--membrane_thickness",help="Initial thickness of membrane in Angstroms (Deafult: 38)")
parser.set_defaults(membrane_thickness=38.0)
parser.add_argument("-wb","--write_bfactors",action="store_true",help="Toggle writing charge at each membrane segment instead of z position.")
parser.set_defaults(write_bfactors=False)
args = parser.parse_args()

#Error checking user inputs

mem_data = [0,0]

try:
	mem_data[1] = float(args.membrane_thickness)-10
except:
	print("ERROR: Could not read value of -mt. Must be a float > 10.")
	exit()
if(mem_data[1] <= 0):
	print("ERROR: Could not read value of -mt. Must be a float > 10.")
	exit()
	

	
try:
	iters = int(args.iters)
except:
	print("ERROR: Could not read value of -ni. Must be an integer >= 0.")
	exit()
if(iters < 0):
	print("ERROR: Could not read value of -ni. Must be an integer >= 0.")
	exit()


try:
	build_system = int(args.build_system)
except:
	print("ERROR: Could not read value of -bd. Must be an integer > -1.")
	exit()
if(build_system < 0):
	print("ERROR: Could not read value of -bd. Must be an integer > -1.")
	exit()



fn = args.file_name

if(not os.path.exists(fn)):
	print("ERROR: Cannot find file: "+fn)
	exit()

orient_dir = args.output

#Setting Martini itp file
martini_file = args.itp_file

if(not os.path.exists(martini_file)):
	print("ERROR: Cannot find file: "+martini_file)
	exit()
	
	
#Creating folders to hold data
if(orient_dir == None):
	if(not os.path.exists("Deformations/")):
		os.mkdir("Deformations/")
	orient_dir = "Deformations/"
else:
	if orient_dir[-1] != "/":
		orient_dir += "/"
	if(not os.path.exists(orient_dir)):
		os.mkdir(orient_dir)
		

locdir = orient_dir+"LocData/"
if(not os.path.exists(locdir)):
	os.mkdir(locdir)
	
memdir = orient_dir+"Membrane_Data/"
if(not os.path.exists(memdir)):
	os.mkdir(memdir)

timer = time.time()
tot_time = timer

#Some useful functions

@jax.jit
def flat_ind(ind1,ind2,max_ind1):
	return ind2*max_ind1+ind1
@jax.jit	

def unflat_ind(indf,max_ind1):
	ind1 = indf%max_ind1
	ind2 = jnp.array((indf-ind1)/max_ind1,dtype=int)
	return ind1,ind2

@jax.jit
def put_into_grid(x,r,n,s):
	cx = x[0]-s[0]
	cy = x[1]-s[1]
	norm_x = cx/r[0]
	norm_y = cy/r[1]
	grid_x = n[0]*norm_x
	grid_y = n[1]*norm_y
	return jnp.array(jnp.floor(grid_x),dtype=int),jnp.array(jnp.floor(grid_y),dtype=int)
	
twopowsixth = jnp.power(2,1/6)

def write_point(points,fn):
	new_file = open(fn,"w")
	count = 0
	for i in points:
		count += 1
		count_str = (6-len(str(count)))*" "+str(count)
		c = "ATOM "+count_str+" BB   DUM	 1	   0.000   0.000  15.000  1.00  0.00" 
		xp = np.format_float_positional(i[0],precision=3)
		yp = np.format_float_positional(i[1],precision=3)
		zp = np.format_float_positional(i[2],precision=3)
		xp += "0"*(3-len((xp.split(".")[1])))
		yp += "0"*(3-len((yp.split(".")[1])))
		zp += "0"*(3-len((zp.split(".")[1])))
		new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:]+"\n"	
		new_file.write(new_c)
	new_file.close()

def write_protein(bfacs,fn,ofn):
	new_file = open(fn,"r")
	n_lines = new_file.readlines()
	new_file.close()
	out_file = open(ofn,"w")
	bc = 0
	for c in n_lines:
		if(len(c) > 46):
			if("[" not in c and c[:4]=="ATOM"):				
				bbp = np.format_float_positional(bfacs[map_to_beads[bc]],precision=3)
				bbp += "0"*(3-len((bbp.split(".")[1])))
				new_c = c[:60]+(" "*(8-len(bbp)))+bbp+c[68:]
				bc+=1
				out_file.write(new_c)
			else:
				out_file.write(c)
		else:
			out_file.write(c)	
	out_file.close()
	
	
#lennard jones function with values <r cutoff	
@jax.jit
def lj(r,min_pos):
	ret_val = 0
	def lmp(r):
		return min_pos*twopowsixth
	def nlmp(r):
		return r
	r = jax.lax.cond(r<min_pos*twopowsixth,lmp,nlmp,r)
	ret_val = 4*(jnp.power((min_pos/r),12)-jnp.power((min_pos/r),6))
	return ret_val
	

def get_loop_inds(n,nx,ny):
	indsx = np.arange(nx-2*n+2)+(n-1)
	indsy = np.arange(ny-2*n+2)+(n-1)
	y0 = np.zeros(nx-2*n+2)+indsy[0]
	yn = np.zeros(nx-2*n+2)+indsy[-1]
	p0 = np.zeros(nx-2*n+2)-1
	pp0 = np.zeros(nx-2*n+2)
	pp0[0] += 1
	pp0[-1] -= 1
	pn = np.zeros(nx-2*n+2)+1
	line1 = np.concatenate((indsx[:,None],y0[:,None],pp0[:,None],pn[:,None]),axis=1)
	line2 = np.concatenate((indsx[:,None],yn[:,None],pp0[:,None],p0[:,None]),axis=1)
	x0 = np.zeros(ny-2*n)+indsx[0]
	xn = np.zeros(ny-2*n)+indsx[-1]
	p0 = np.zeros(ny-2*n)-1
	pp0 = np.zeros(ny-2*n)
	pn = np.zeros(ny-2*n)+1
	line3 = np.concatenate((x0[:,None],indsy[1:-1,None],pn[:,None],pp0[:,None]),axis=1)
	line4 = np.concatenate((xn[:,None],indsy[1:-1,None],p0[:,None],pp0[:,None]),axis=1)
	return np.concatenate((line1,line2,line3,line4))
	
def get_all_loop_inds(k,nx,ny):
	all_inds = np.empty((0,4))
	for i in range(k):
		all_inds = np.concatenate((get_loop_inds(i+1,nx,ny),all_inds))
	return all_inds

def get_new_zs(new_zs,loop_inds):
	nx = new_zs.shape[0]
	ny = new_zs.shape[1]
	def loop1(new_zs,ind):
		inder = jnp.array(loop_inds[ind],dtype=int)
		ix = inder[0]
		iy = inder[1]
		inderm1 = inder.at[:2].set(inder[:2]+inder[2:])
		ixl = inderm1[0]
		iyl = inderm1[1]
		new_zs = new_zs.at[ix,iy,0].set(new_zs[ix,iy,0]+new_zs[ixl,iyl,0])
		return new_zs , ind
	new_zs,_ = jax.lax.scan(loop1,new_zs,jnp.arange(loop_inds.shape[0]))
	return new_zs
	
	
	
	
		
	
#Sigmoid function
@jax.jit	
def sj(x,grad):
	ret_val = -x*grad
	def overflow_pos(ret_val):
		ret_val = 100.0
		return ret_val
	def overflow_neg(ret_val):
		ret_val = -100.0
		return ret_val
	def not_overflow(ret_val):
		return ret_val
	
	ret_val = jax.lax.cond(-x*grad>100,overflow_pos,not_overflow,ret_val)
	ret_val = jax.lax.cond(-x*grad<-100,overflow_neg,not_overflow,ret_val)
	return 1.0/(1.0+jnp.exp(ret_val))
	

#Two variations of the mean field function
@jax.jit
def smj(x,grad,bead_num,mem_structure):
	l1min = (W_B_mins[bead_num]+LH1_B_mins[bead_num]-jnp.abs(W_B_mins[bead_num]-LH1_B_mins[bead_num]))/2
	l2min = (W_B_mins[bead_num]+LH2_B_mins[bead_num]-jnp.abs(W_B_mins[bead_num]-LH2_B_mins[bead_num]))/2
	l3min = (W_B_mins[bead_num]+LH3_B_mins[bead_num]-jnp.abs(W_B_mins[bead_num]-LH3_B_mins[bead_num]))/2
	bd1 = (1.0-sj(x-mem_structure[0],grad))*W_B_mins[bead_num]+sj(x-mem_structure[0],grad)*l1min
	bd2 = (1.0-sj(x-mem_structure[1],grad))*bd1+sj(x-mem_structure[1],grad)*l2min
	bd3 = (1.0-sj(x-mem_structure[2],grad))*bd2+sj(x-mem_structure[2],grad)*l3min
	bd4 = (1.0-sj(x-mem_structure[3],grad))*bd3+sj(x-mem_structure[3],grad)*LT1_B_mins[bead_num]
	bd5 = (1.0-sj(x-mem_structure[4],grad))*bd4+sj(x-mem_structure[4],grad)*l3min
	bd6 = (1.0-sj(x-mem_structure[5],grad))*bd5+sj(x-mem_structure[5],grad)*l2min
	bd7 = (1.0-sj(x-mem_structure[6],grad))*bd6+sj(x-mem_structure[6],grad)*l1min
	bd8 = (1.0-sj(x-mem_structure[7],grad))*bd7+sj(x-mem_structure[7],grad)*W_B_mins[bead_num]
	return bd8-W_B_mins[bead_num]


@jax.jit
def smjh(x,grad,bead_num,mem_structure):
	bd1 = (1.0-sj(x-mem_structure[0],grad))*W_B_mins[bead_num]+sj(x-mem_structure[0],grad)*LH1_B_mins[bead_num]
	bd2 = (1.0-sj(x-mem_structure[1],grad))*bd1+sj(x-mem_structure[1],grad)*LH2_B_mins[bead_num]
	bd3 = (1.0-sj(x-mem_structure[2],grad))*bd2+sj(x-mem_structure[2],grad)*LH3_B_mins[bead_num]
	bd4 = (1.0-sj(x-mem_structure[3],grad))*bd3+sj(x-mem_structure[3],grad)*LT1_B_mins[bead_num]
	bd5 = (1.0-sj(x-mem_structure[4],grad))*bd4+sj(x-mem_structure[4],grad)*LH3_B_mins[bead_num]
	bd6 = (1.0-sj(x-mem_structure[5],grad))*bd5+sj(x-mem_structure[5],grad)*LH2_B_mins[bead_num]
	bd7 = (1.0-sj(x-mem_structure[6],grad))*bd6+sj(x-mem_structure[6],grad)*LH1_B_mins[bead_num]
	bd8 = (1.0-sj(x-mem_structure[7],grad))*bd7+sj(x-mem_structure[7],grad)*W_B_mins[bead_num]
	return bd8-W_B_mins[bead_num]
	
#Creates a list containing the membrane structure for a given membrane thickness
def get_mem_struc(mt):
	memt_tails = mt
	memt_heads = 10.0
	memt_total = memt_tails+memt_heads
	
	h1_w = memt_heads/6.0
	h2_w = memt_heads/6.0
	h3_w = memt_heads/6.0
	
	l_w = memt_tails
	meml = -l_w/2.0 -h1_w -h2_w-h3_w
	mem_structure = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
	return mem_structure

#A fucntion that reads a martini file to get interaction strengths
def get_int_strength(bead_1,bead_2,martini_file):
	if(bead_2 == "GHOST"):
		return 0
	string = " "*(6-len(bead_1))+bead_1+" "*(6-len(bead_2))+bead_2
	string2 = " "*(6-len(bead_2))+bead_2+" "*(6-len(bead_1))+bead_1
	mfile = open(martini_file,"r")
	content = mfile.readlines()
	for i,line in enumerate(content):
		if(string in line or string2 in line):
			return -float(line[32:45])
	
#Gets the interations between protein beads and the enviroment. This is used to create the mean fields		
def get_mem_def(martini_file):
	#We use interactions strengths from martini using a POPE(Q4p)/POPG(P4)/POPC(Q1)? lipid as a template
	W_B_mins = jnp.array([get_int_strength("W",Beadtype(i).name,martini_file)-get_int_strength("W",Beadtype(i).name,martini_file) for i in range(23)])
	LH1_B_mins = jnp.array([get_int_strength("P4",Beadtype(i).name,martini_file)-get_int_strength("W",Beadtype(i).name,martini_file) for i in range(23)])
	LH2_B_mins = jnp.array([get_int_strength("Q5",Beadtype(i).name,martini_file)-get_int_strength("W",Beadtype(i).name,martini_file) for i in range(23)])
	LH3_B_mins = jnp.array([get_int_strength("SN4a",Beadtype(i).name,martini_file)-get_int_strength("W",Beadtype(i).name,martini_file) for i in range(23)])
	LH4_B_mins = jnp.array([get_int_strength("N4a",Beadtype(i).name,martini_file)-get_int_strength("W",Beadtype(i).name,martini_file) for i in range(23)])
	LT1_B_mins = jnp.array([get_int_strength("C1",Beadtype(i).name,martini_file)-get_int_strength("W",Beadtype(i).name,martini_file) for i in range(23)])
	LT2_B_mins = jnp.array([get_int_strength("C4h",Beadtype(i).name,martini_file)-get_int_strength("W",Beadtype(i).name,martini_file) for i in range(23)])
	Charge_B_mins =jnp.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,-1,-1,0,0,0,0,0],dtype="float64")
							# 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5  6  7 8 9 0 1 2							   # #
	return (W_B_mins,LH1_B_mins,LH2_B_mins,LH3_B_mins,LH4_B_mins,LT1_B_mins,LT2_B_mins,Charge_B_mins)
	
#Gets the n closest neighbours in a grid
def get_nbh_inds(inder,nx,ny,n):
	ix,iy = unflat_ind(inder,nx)
	
	defs = jnp.arange(2*n+1)-n
	nbh_inds = jnp.zeros(defs.shape[0]*defs.shape[0])
	def l1(nbh_inds,ind):
		ind1 = ind
		def l2(nbh_inds,ind):
			new_x = ix+defs[ind1]
			new_y = iy+defs[ind]
			def wrap_up(new_ind,ind_x):
				return new_ind+ind_x
			def wrap_down(new_ind,ind_x):
				return new_ind-ind_x
			def nothing(new_ind,ind_x):
				return new_ind
				
			new_x = jax.lax.cond(new_x>=nx,wrap_down,nothing,new_x,nx)
			new_y = jax.lax.cond(new_y>=ny,wrap_down,nothing,new_y,ny)
			new_x = jax.lax.cond(new_x<0,wrap_up,nothing,new_x,nx)
			new_y = jax.lax.cond(new_y<0,wrap_up,nothing,new_y,ny)
			
			nbh_inds = nbh_inds.at[ind1*defs.shape[0]+ind].set(flat_ind(new_x,new_y,nx))
			return nbh_inds,ind
		nbh_inds,_=jax.lax.scan(l2,nbh_inds,jnp.arange(defs.shape[0]))
		return nbh_inds,ind
	nbh_inds,_=jax.lax.scan(l1,nbh_inds,jnp.arange(defs.shape[0]))
	return nbh_inds
	
#Writes the deformations pdb file		
def write_pdb(start_z,grid_in,grid_ban,nx,ny,exx,exy,gc,defo_dir,locdir,memdir,bfacs,memdata,rot_mat_inv):
	BanGrid = np.zeros((nx,ny,2))
	Data = np.zeros((nx,ny,2))
	new_file = open(defo_dir+"Deformations.pdb","w")
	grid_in = np.array(grid_in)
	count = 0
	xs = np.linspace(-exx,exx,nx)
	ys = np.linspace(-exy,exy,ny)
	np.savez(locdir+"Xlin.npz",xs)
	np.savez(locdir+"Ylin.npz",ys)
	np.savez(memdir+"Xlin.npz",xs)
	np.savez(memdir+"Ylin.npz",ys)
	for i in range(nx):
		for j in range(ny):
			if(grid_ban[i,j] == 1):
				ms = get_mem_struc(grid_in[i,j,1])
				ms_av = get_mem_struc(grid_in[0,0,1])
				z_pluss = [ms[0],ms[7]]
				z_pluss_av = [ms_av[0],ms_av[7]]
				for nn,zz in enumerate(z_pluss):
					pm1 = (nn-0.5)*2
					zp = zz+start_z/5
					z_dev = np.mean(grid_in[0,:,0]+z_pluss_av[nn]+start_z/5)
					zp_vpos = jnp.array(((grid_in[i,j,0]+zp+vext)/(2*vext))*vert_grid_size,dtype=int)
					zp_vposc = jnp.array(((grid_in[i,j,0]+vext)/(2*vext))*vert_grid_size,dtype=int)
					if(zp_vpos < zp_vposc):
						lb = zp_vpos
						rb = zp_vposc
					else:
						lb = zp_vposc
						rb = zp_vpos
					lenny = rb-lb
					if(lb > 49):
						lb = 49
					if(rb > 49):
						rb=49
					if(lb<0):
						lb=0
					if(rb<0):
						rb=0
					covp = jnp.sum(bgrid_o[i,j,lb:rb])/lenny
					if(covp > 0.6):
						BanGrid[i,j,nn] = 1
						Data[i,j,nn] = grid_in[i,j,0]+zp
						if(bfacs):
							bbp = np.format_float_positional(gc[i,j,nn],precision=3)
						else:
							bbp = np.format_float_positional((grid_in[i,j,0]+zp-z_dev)*pm1,precision=3)						
						bbp += "0"*(3-len((bbp.split(".")[1])))
						count += 1
						count_str = (6-len(str(count)))*" "+str(count)
						c = "ATOM "+count_str+" BB   DUM Z   1	   0.000   0.000  15.000  1.00  0.00" 
												
						pos = np.array([xs[i],ys[j],0])
						pos = np.dot(rot_mat_inv,pos)
						xp = np.format_float_positional(pos[0],precision=3)
						yp = np.format_float_positional(pos[1],precision=3)
						zp = np.format_float_positional(grid_in[i,j,0]+zp,precision=3)
						xp += "0"*(3-len((xp.split(".")[1])))
						yp += "0"*(3-len((yp.split(".")[1])))
						zp += "0"*(3-len((zp.split(".")[1])))
						new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:60]+(" "*(8-len(bbp)))+bbp+c[66:]+"\n"	
						new_file.write(new_c)
	new_file.close()
	np.savez(locdir+"BotBan.npz",BanGrid[:,:,0])
	np.savez(locdir+"TopBan.npz",BanGrid[:,:,1])
	np.savez(locdir+"BotData.npz",Data[:,:,0])
	np.savez(locdir+"TopData.npz",Data[:,:,1])
	return BanGrid[:,:,0],BanGrid[:,:,1]
	
#calcualtes the eleectrostatic force using below
def calc_esf(start_z,grid_in,grid_ban,nx,ny,exx,exy,cposes,ctypes):
	grid_in = np.array(grid_in)
	grid_charge = np.zeros((nx,ny,2))
	count = 0
	xs = np.linspace(-exx,exx,nx)
	ys = np.linspace(-exy,exy,ny)
	for i in range(nx):
		for j in range(ny):
			if(grid_ban[i,j] == 1):
				ms = get_mem_struc(grid_in[i,j,1])
				z_pluss = [ms[0],ms[7]]
				for nn,zz in enumerate(z_pluss):
					zp = zz+start_z/5
					x_pos = xs[i]
					y_pos = ys[j]
					z_pos = grid_in[i,j,0]+zp
					grid_charge[i][j][nn] = calc_esf_p(jnp.array([x_pos,y_pos,z_pos]),cposes,ctypes)
	return grid_charge
		
#calcues the electrostatic force
@jax.jit			
def calc_esf_p(pos,cposes,ctypes):
	ctypes = jnp.array(ctypes,dtype=int)
	charge_const = 92.64
	Charge_B_mins =jnp.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,-1,-1,0,0,0,0,0],dtype="float64")
	def esf_loop(carry,ind):
		charge_cont = Charge_B_mins[ctypes[ind]]*charge_const/jnp.linalg.norm(pos-cposes[ind])
		carry += charge_cont
		return carry,ind
	tot_c,_ = jax.lax.scan(esf_loop,0,jnp.arange(cposes.shape[0]))
	return tot_c		
						
					

	
		
#Creating a helper class that deals with loading the PDB files
PDB_helper_test = ori.PDB_helper(fn,False,0,np.array([]),args.no_cavity_surface,True)

#Loading PDB
pos_mean = PDB_helper_test.load_pdb()


#Getting surface
print("Getting surface residues...")
timer = time.time()
jax.block_until_ready(PDB_helper_test.get_surface())
PDB_helper_test.recenter_bvals()
data = PDB_helper_test.get_data()
int_data = get_mem_def(martini_file)
print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")

#storing important data
surf_poses = data[2]+pos_mean
poses = data[5]+pos_mean
surf_bvals = data[4]
bead_types = data[3]
surface_no = data[0]
map_to_beads = data[9]


#Rotates the protein to minimise the overall size of the grid. This improves efficiency for wide proteins.
print("Optimising grid...")
timer = time.time()

angs = jnp.linspace(0,jnp.pi,200)
def min_extent(carry,ind):
	nang = angs[ind]
	rot_mat = jnp.array([[jnp.cos(nang),-jnp.sin(nang),0],[jnp.sin(nang),jnp.cos(nang),0],[0,0,1]])
	test_surf_poses = jnp.dot(rot_mat,poses.T).T
	x_ext = jnp.maximum(-jnp.min(test_surf_poses[:,0]),jnp.max(test_surf_poses[:,0]))
	y_ext = jnp.maximum(-jnp.min(test_surf_poses[:,1]),jnp.max(test_surf_poses[:,1]))
	def lower(carry):
		carry[0] = nang
		carry[1] = x_ext*y_ext
		return carry
	def higher(carry):
		return carry
	carry = jax.lax.cond(carry[1] > x_ext*y_ext,lower,higher,carry)
	return carry,ind

carry,_ = jax.lax.scan(min_extent,[0.0,jnp.inf],jnp.arange(200))
nang = carry[0]
rot_mat = jnp.array([[jnp.cos(nang),-jnp.sin(nang),0],[jnp.sin(nang),jnp.cos(nang),0],[0,0,1]])
rot_mat_inv = jnp.array([[jnp.cos(-nang),-jnp.sin(-nang),0],[jnp.sin(-nang),jnp.cos(-nang),0],[0,0,1]])
surf_poses = jnp.dot(rot_mat,surf_poses.T).T
poses = jnp.dot(rot_mat,poses.T).T


print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")
timer = time.time()

#Getting the chared residues for charge calculations later
print("Getting charged residues...")

surf_cposes = jnp.zeros((surf_poses.shape[0],3))
cbead_types = jnp.zeros(bead_types.shape[0])

def cloop(carry,ind):
	def charged(carry):
		new0 = carry[0].at[carry[2]].set(surf_poses[ind])
		new1 = carry[1].at[carry[2]].set(bead_types[ind])
		new2 = carry[2] + 1
		return (new0,new1,new2)
	def ncharged(carry):
		return carry
	carry = jax.lax.cond((bead_types[ind] == 12)+(bead_types[ind] == 13)+(bead_types[ind] == 16)+(bead_types[ind] == 17),charged,ncharged,carry)
	return carry,ind

carry = (surf_cposes,cbead_types,0)
carry,_ = jax.lax.scan(cloop,carry,jnp.arange(surf_poses.shape[0]))
cposes = carry[0][:carry[2]]
ctypes = carry[1][:carry[2]]
	
print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")


#setting up variables used later

vert_grid_size = 50
vext = 50
vlin = jnp.linspace(-vext,vext,vert_grid_size)

indexer = np.nonzero((np.array(surf_poses[:,2]) < 50)*(np.array(surf_poses[:,2] > -50)))

surf_poses = surf_poses[indexer]
surf_bvals = surf_bvals[indexer]
bead_types = bead_types[indexer]

indexer2 = np.nonzero((np.array(poses[:,2]) < 50)*(np.array(poses[:,2] > -50)))

all_poses = poses.copy()
poses = poses[indexer2]

cutoff = int(args.cutoff)

x_ext = max(-jnp.min(surf_poses[:,0]),jnp.max(surf_poses[:,0]))+20
y_ext = max(-jnp.min(surf_poses[:,1]),jnp.max(surf_poses[:,1]))+20

grid_den = float(args.grid_density)

rad = int(4.0/grid_den)

grid_size_x = int(2*x_ext/grid_den)
grid_size_y = int(2*y_ext/grid_den)

loop_inds = jnp.array(get_all_loop_inds(12,grid_size_x,grid_size_y))
loop_inds_end = np.array(get_loop_inds(1,grid_size_x,grid_size_y),dtype=int)

nxx = jnp.array(x_ext/cutoff,dtype=int)
nyy = jnp.array(y_ext/cutoff,dtype=int)

no_beads = surf_poses.shape[0]

#Putting beads into bins for more efficient allocation of beads to membrane segments later on
print("Binning surface... (This could take a short while)")
timer = time.time()
@jax.jit
def allo_bead(surf_poses):
	no_beads = surf_poses.shape[0]
	big_grid = jnp.zeros((grid_size_x,grid_size_y,no_beads+1),dtype=int)-1
	big_grid = big_grid.at[:,:,no_beads].set(0)
	def allocacte_beads(big_grid,ind):
		bind = ind
		x_pos = surf_poses[bind][0]
		y_pos = surf_poses[bind][1]
		xgrid,ygrid = put_into_grid([x_pos,y_pos],[2*x_ext,2*y_ext],[nxx,nyy],[-x_ext,-y_ext])
		big_grid = big_grid.at[xgrid,ygrid,big_grid[xgrid,ygrid,-1]].set(bind)
		big_grid = big_grid.at[xgrid,ygrid,-1].set(big_grid[xgrid,ygrid,-1]+1)
		return big_grid,ind
	big_grid,_ = jax.lax.scan(allocacte_beads,big_grid,jnp.arange(no_beads))
	return big_grid
	
big_grid = allo_bead(surf_poses)
big_gridb = allo_bead(poses)

max_beads = jnp.max(big_grid[:,:,-1])

print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")
timer = time.time()

#Allocating beads to membrane segments for use when calculating the potential. This takes some time and so is pre-calculated.
print("Allocating beads to membrane segments... (This could take a short while)")

xs = jnp.linspace(-x_ext,x_ext,grid_size_x)
ys = jnp.linspace(-y_ext,y_ext,grid_size_y)
indxes = jnp.zeros((grid_size_x,grid_size_y,2),dtype=int)
def indloop1(indxes,ind):
	def indloop2(indxes,ind2):
		indxes = indxes.at[ind,ind2].set(jnp.array([ind,ind2]))
		return indxes,ind2
	indxes,_ = jax.lax.scan(indloop2,indxes,jnp.arange(grid_size_y))
	return indxes,ind
indxes,_=jax.lax.scan(indloop1,indxes,jnp.arange(grid_size_x))

indxes = indxes.reshape((grid_size_x*grid_size_y,2))

def allo_grid_vmap(cutoff_val,big_grid,surf_poses):
	max_beads = jnp.max(big_grid[:,:,-1])
	@partial(jax.vmap,in_axes=0)
	def agrid_vmap(inds):
		grid_p = jnp.zeros((no_beads+1,2),dtype=float)-1
		grid_p = grid_p.at[no_beads,:].set(0)
		ind2 = inds[0]
		ind3 = inds[1]
		indx,indy = put_into_grid([xs[ind2],ys[ind3]],[2*x_ext,2*y_ext],[nxx,nyy],[-x_ext,-y_ext])
		sq = jnp.array([[0,0],[-1,0],[1,0],[0,-1],[0,1],[-1,1],[-1,-1],[1,1],[1,-1]],dtype=int)
		def allocate_grid3(grid_p,ind):
			def allocate_grid4(grid_p,sind):
				bgind = big_grid[(indx+sq[sind,0])%grid_size_x,(indy+sq[sind,1])%grid_size_y][ind]
				def close(grid_p,dist):
					indeex = jnp.array(grid_p[-1,0],dtype=int)
					grid_p = grid_p.at[indeex,0].set(bgind)
					grid_p = grid_p.at[indeex,1].set(dist)
					grid_p = grid_p.at[-1,0].set(grid_p[-1,0]+1)
					return grid_p
				def far(grid_p,dist):
					return grid_p			
				def is_bead(grid_p):
					pos = surf_poses[bgind][0:2]
					grid_pos = jnp.array([xs[ind2],ys[ind3]])
					dist = jnp.linalg.norm(pos-grid_pos)
					grid_p = jax.lax.cond(dist < cutoff_val, close,far,grid_p,dist)
					return grid_p
				def isn_bead(grid_p):
					return grid_p
				grid_p= jax.lax.cond(bgind != -1,is_bead,isn_bead,grid_p)
				return grid_p,sind
			grid_p,_ = jax.lax.scan(allocate_grid4,grid_p,jnp.arange(9))
			return grid_p,ind
		grid_p,_ = jax.lax.scan(allocate_grid3,grid_p,jnp.arange(max_beads))
		return grid_p
	grid_p = agrid_vmap(indxes)
	grid = grid_p.reshape((grid_size_x,grid_size_y,no_beads+1,2))
	return grid

grid = allo_grid_vmap(cutoff,big_grid,surf_poses).block_until_ready()
grid_block = allo_grid_vmap(3.0,big_gridb,poses)

print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")
timer = time.time()

#Creating a smooth function that indicates if a point is with the protein or not.
print("Blocking membrane inside protein...")

max_beads_block = jnp.array(jnp.max(grid_block[:,:,-1,0]),dtype=int)
bgrid = jnp.zeros((grid_size_x,grid_size_y,vert_grid_size))+1

def gblock1(bgrid,ind):
	ind2 = ind
	def gblock2(bgrid,ind):
		ind3 = ind
		def bloop(bgrid,ind):
			bind = jnp.array(grid_block[ind2,ind3,ind,0],dtype=int)
			def mins(bgrid):
				return bgrid
			def nmins(bgrid):
				atom_pos = poses[bind,2]
				indap = jnp.array(((atom_pos+vext)/(2*vext))*vert_grid_size,dtype=int)
				bgrid = bgrid.at[ind2,ind3,indap].set(0)
				bgrid = bgrid.at[ind2,ind3,indap-1].set(0)
				bgrid = bgrid.at[ind2,ind3,indap+1].set(0)
				return bgrid
			bgrid = jax.lax.cond(bind != -1,nmins,mins,bgrid)
			return bgrid,ind
		bgrid,_ = jax.lax.scan(bloop,bgrid,jnp.arange(max_beads_block))
		return bgrid,ind
	bgrid,_ = jax.lax.scan(gblock2,bgrid,jnp.arange(grid_size_y))
	return bgrid,ind
bgrid,_ = jax.lax.scan(gblock1,bgrid,jnp.arange(grid_size_x))

			
mem_structure = get_mem_struc(mem_data[1])

def accessable(grid):
	ngrid = jnp.zeros_like(grid)
	def aloop1(ngrid,ind1):
		def aloop2(ngrid,ind2):
			t1 = jnp.sum(jnp.array([grid[ind1,ind2],grid[ind1-1,ind2],grid[ind1-1,ind2-1],grid[ind1,ind2-1]]))
			t2 = jnp.sum(jnp.array([grid[ind1,ind2],grid[ind1+1,ind2],grid[ind1+1,ind2-1],grid[ind1,ind2-1]]))
			t3 = jnp.sum(jnp.array([grid[ind1,ind2],grid[ind1-1,ind2],grid[ind1-1,ind2+1],grid[ind1,ind2+1]]))
			t4 = jnp.sum(jnp.array([grid[ind1,ind2],grid[ind1+1,ind2],grid[ind1+1,ind2+1],grid[ind1,ind2+1]]))
			def good(ngrid):
				ngrid = ngrid.at[ind1,ind2].set(1)
				return ngrid
			def bad(ngrid):
				return ngrid
			ngrid = jax.lax.cond(jnp.logical_or(jnp.logical_or(t1==4,t2==4),jnp.logical_or(t3==4,t4==4)),good,bad,ngrid)
			return ngrid,ind2
		ngrid,_= jax.lax.scan(aloop2,ngrid,jnp.arange(ngrid.shape[1]))
		return ngrid,ind1
	ngrid,_ = jax.lax.scan(aloop1,ngrid,jnp.arange(ngrid.shape[0]))
	return ngrid
				
grid_ban = jnp.zeros((grid_size_x,grid_size_y),dtype=int)+1
@jax.jit
def fill_mid(grid_ban,tol):
	rdic_carry = jnp.zeros((grid_size_x*grid_size_y,3))
	numer = grid_size_x*grid_size_y
	rdic_carry = rdic_carry.at[0,0].set(1)
	def remove_disc(rdic_carry,ind):
		def check(rdic_carry):
			rdic_carry = rdic_carry.at[ind,2].set(1)
			def is_block(rdic_carry):
				return rdic_carry
			def is_no_block(rdic_carry):
				nb_inds = jnp.array(get_nbh_inds(ind,grid_size_x,grid_size_y,1),dtype=int)
				rdic_carry = rdic_carry.at[nb_inds,0].set(rdic_carry[nb_inds,0]+1)
				rdic_carry = rdic_carry.at[ind,1].set(1)
				return rdic_carry
			rdic_carry = jax.lax.cond(grid_ban[unflat_ind(ind,grid_size_x)]>tol,is_block,is_no_block,rdic_carry)		
			return rdic_carry
		def no_check(rdic_carry):
			return rdic_carry
		rdic_carry = jax.lax.cond((rdic_carry[ind,0]>0)*(rdic_carry[ind,2]==0),check,no_check,rdic_carry)		
		return rdic_carry,ind
	
	rdic_carry,_ = jax.lax.scan(remove_disc,rdic_carry,jnp.arange(numer))
	rdic_carry,_ = jax.lax.scan(remove_disc,rdic_carry,-jnp.arange(numer)+(numer-1))
	transp_arange = jnp.arange(numer).reshape(grid_size_x,grid_size_y)
	transp_arange = transp_arange.T
	transp_arange = transp_arange.reshape(numer)
	rdic_carry,_ = jax.lax.scan(remove_disc,rdic_carry,transp_arange)
	rdic_carry,_ = jax.lax.scan(remove_disc,rdic_carry,-transp_arange+(numer-1))
	return rdic_carry[:,2].reshape(grid_size_y,grid_size_x).T

bgrid_o = jnp.zeros_like(bgrid)

def fill_out_all(carry,ind):
	bgrid = carry[0]
	bgrid_o = carry[1]
	tempb = fill_mid(1-accessable(bgrid[:,:,ind]),0.5)
	bgrid_o = bgrid_o.at[:,:,ind].set(tempb)
	bgrid = bgrid.at[:,:,ind].set(fill_mid(1-tempb,0.5))
	return (bgrid,bgrid_o),ind
	
carryb,_ = jax.lax.scan(fill_out_all,(bgrid,bgrid_o),jnp.arange(vert_grid_size))
bgrid = carryb[0]
bgrid_o = carryb[1]	

print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")
timer = time.time()

#This function indicates if grid point (i,j) and height z is within the protein. 
@jax.jit
def bgrid_weight(i,j,z):
	zind = ((z+vext)/(2*vext))*vert_grid_size
	zind_l = jnp.array(jnp.floor(zind),dtype=int)-5
	tot = 0
	
	def above(zindy):
		return vert_grid_size-1
	def below(zindy):
		return 0
	def nab(zindy):
		return jnp.array(zindy,dtype=int)
			
	def bwloop(tot,ind):
		zind_u = zind_l+ind
		zind_u = jax.lax.cond(zind_u > vert_grid_size-1,above,nab,zind_u)
		zind_u = jax.lax.cond(zind_u < 0,below,nab,zind_u)
		tot += bgrid[i,j,zind_u]
		
		return tot,ind
	tot,_ = jax.lax.scan(bwloop,tot,jnp.arange(10))
	
	zind_u = jax.lax.cond(zind_l+10 > vert_grid_size-1,above,nab,zind_l+10)
	zind_u = jax.lax.cond(zind_l+10 < 0,below,nab,zind_l+10)
	
	zind_l = jax.lax.cond(zind_l > vert_grid_size-1,above,nab,zind_l)
	zind_l = jax.lax.cond(zind_l < 0,below,nab,zind_l)
	
	tot += (bgrid[i,j,zind_u]-bgrid[i,j,zind_l])*(zind-zind_l-5)
	return (1-jnp.cos(jnp.pi*(tot/10)))/2.0
	
max_beads = jnp.array(jnp.max(grid[:,:,-1,0]),dtype=int)
W_B_mins = int_data[0]
LH1_B_mins = int_data[1]
LH2_B_mins = int_data[2]
LH3_B_mins = int_data[3]
LH4_B_mins = int_data[4]
LT1_B_mins = int_data[5]
LT2_B_mins = int_data[6]
Charge_B_mins =int_data[7]

#Prevents gaps between protein and membrane
def intbgrid(ind2,ind3,zt,mt,zd):
	mem_structure = get_mem_struc(mt)
	z_disp = zt+zd
	logs = jnp.linspace(mem_structure[0],mem_structure[-1],10)
	sumer = 0
	def z_range(sumer,ind):
		zr = logs[ind]+z_disp
		sumer += bgrid_weight(ind2,ind3,zr)
		return sumer, ind
	sumer,_ =jax.lax.scan(z_range,sumer,jnp.arange(10))
	return (sj(-sumer+4,3))*(-sumer-200)/5.0
		


#Potential function. If surface bead or membrane segment is within the protein it's contribution is ignored
def get_pot(ind2,ind3,zt,mt,zd):
	no_beads = grid[ind2,ind3,-1,0]
	z_disp = zt+zd
	def calc_grid3(tot,ind):
		mem_structure = get_mem_struc(mt)
		bind = jnp.array(grid[ind2,ind3,ind,0],dtype=int)
		dist = grid[ind2,ind3,ind,1]
		def nomin(tot):
			z_pos = surf_poses[bind][2]-z_disp
			xy_pos = surf_poses[bind][0:2]
			indx,indy = put_into_grid([xy_pos[0],xy_pos[1]],[2*x_ext,2*y_ext],[nxx,nyy],[-x_ext,-y_ext])
			bead_num = bead_types[bind]
			disterz = jnp.sqrt(dist*dist)
			def close(tot):
				def upper(tot):
					tot = tot.at[0].set(tot[0]+smj(z_pos,0.5,bead_num,mem_structure)*-1*lj(disterz,8.0)*bgrid_weight(ind2,ind3,z_pos)*bgrid_weight(indx,indy,z_pos))
					return tot
				def lower(tot):
					tot = tot.at[1].set(tot[1]+smj(z_pos,0.5,bead_num,mem_structure)*-1*lj(disterz,8.0)*bgrid_weight(ind2,ind3,z_pos)*bgrid_weight(indx,indy,z_pos))
					return tot
				tot = jax.lax.cond(z_pos>0,upper,lower,tot)
				return tot
			def far(tot):
				return tot
			tot = jax.lax.cond((dist<float(args.cutoff))*(dist > 0),close,far,tot)
			return tot
		def miner(tot):
			return tot
		tot = jax.lax.cond(bind > -1e-1,nomin,miner,tot)
		return tot,ind
	def edge():
		return (zt*zt)*1e-7+((mt-mem_data[1])*(mt-mem_data[1]))*1e-7
	def nedge():
		return (zt*zt)*1e-6+((mt-mem_data[1])*(mt-mem_data[1]))*1e-6
	s_val = jax.lax.cond(jnp.logical_or(jnp.logical_or(ind3==0,ind3==grid.shape[1]-1),jnp.logical_or(ind2==0,ind2==grid.shape[0]-1)),edge,nedge)
	tot,_ = jax.lax.scan(calc_grid3,jnp.array([s_val,s_val]),jnp.arange(max_beads))	
	tot += intbgrid(ind2,ind3,zt,mt,zd)
	return tot,no_beads
		
#Calculates contacts with head groups and acyl tails		
def calc_contacts(new_zs,poses,zdisp,xs,ys):
	bfacsH = jnp.zeros(poses.shape[0])
	bfacsT = jnp.zeros(poses.shape[0])
	mindd = 3
	def ccloop1(bfacs,ind1):
		def ccloop2(bfacs,ind2):
			mt = new_zs[ind1,ind2,1]
			z = new_zs[ind1,ind2,0]
			pos1 = jnp.array([xs[ind1],ys[ind2],z-mt/2.0-4+zdisp])
			pos2 = jnp.array([xs[ind1],ys[ind2],z+mt/2.0+4+zdisp])
			def ccloop3(bfacs,ind3):
				bfacsH = bfacs[0]
				bfacsT = bfacs[1]
				posprot = poses[ind3]
				dist1 = jnp.linalg.norm(posprot-pos1)/10
				dist2 = jnp.linalg.norm(posprot-pos2)/10
				
				bfacsH = bfacsH.at[ind3].set(bfacsH[ind3]+jnp.exp(-jnp.max(jnp.array([dist1*dist1,mindd])))*bgrid_weight(ind1,ind2,z-mt/2.0+4+zdisp)+jnp.exp(-jnp.max(jnp.array([dist2*dist2,mindd])))*bgrid_weight(ind1,ind2,z-mt/2.0-4+zdisp))
				zscan = jnp.linspace(pos1[2],pos2[2],40)
				def ccloop4(bfacsT,ind4):
					pos3 = pos1.copy()
					pos3 = pos3.at[2].set(zscan[ind4])
					dist3 = jnp.linalg.norm(posprot-pos3)/10
					bfacsT = bfacsT.at[ind3].set(bfacsT[ind3]+jnp.exp(-jnp.max(jnp.array([dist3*dist3,mindd])))*bgrid_weight(ind1,ind2,zscan[ind4]))
					return bfacsT,ind4
				bfacsT,_ = jax.lax.scan(ccloop4,bfacsT,jnp.arange(40))
				return (bfacsH,bfacsT),ind3
			bfacs,_ = jax.lax.scan(ccloop3,bfacs,jnp.arange(poses.shape[0]))
			return bfacs,ind2
		bfacs,_ = jax.lax.scan(ccloop2,bfacs,jnp.arange(new_zs.shape[1]))
		return bfacs,ind1
	bfacs,_ = jax.lax.scan(ccloop1,(bfacsH,bfacsT),jnp.arange(new_zs.shape[0]))
	return bfacs
				
PhtoW = get_int_strength("C1","W",martini_file)-get_int_strength("C1","W",martini_file)
PhtoPh = get_int_strength("C1","C1",martini_file)-get_int_strength("C1","W",martini_file)
PhtoH = get_int_strength("P4","C1",martini_file)-get_int_strength("C1","W",martini_file)
HtoH = get_int_strength("P4","P4",martini_file)-get_int_strength("C1","W",martini_file)
HtoW = get_int_strength("P4","W",martini_file)-get_int_strength("C1","W",martini_file)
WtoW = get_int_strength("W","W",martini_file)-get_int_strength("C1","W",martini_file)


#A smooth function that describes the membrane seegment (0 water, 1 head, 2 acyl tails)
@jax.jit
def lip_sm(x,grad,lower,upper):
	bd1 = (1.0-sj(x-lower+5,grad))*2+sj(x-lower+5,grad)*1
	bd2 = (1.0-sj(x-lower,grad))*bd1+sj(x-lower,grad)*0
	bd3 = (1.0-sj(x-upper,grad))*bd2+sj(x-upper,grad)*1
	bd4 = (1.0-sj(x-upper-5,grad))*bd3+sj(x-upper-5,grad)*2
	return bd4
	
int_mat = jnp.array([[PhtoPh,PhtoH,PhtoW],[PhtoH,HtoH,HtoW],[PhtoW,HtoW,WtoW]])

#This function calculates the potential between two membrane segments via 3d linear interpolations using membrane descriptions as above
@jax.jit
def rpot_col(new_zs,inda1,indb1,inda2,indb2,dist,zd,k1):
	dist = jnp.sqrt((inda1-inda2)*(inda1-inda2)+(indb1-indb2)*(indb1-indb2))
	def close():
	
	
		posi1 = new_zs[inda1,indb1]
		upper1 = zd+posi1[0]+0.5*posi1[1]
		lower1 = zd+posi1[0]-0.5*posi1[1]
		
		posi2 = new_zs[inda2,indb2]
		upper2 = zd+posi2[0]+0.5*posi2[1]
		lower2 = zd+posi2[0]-0.5*posi2[1]
		
		lupper = jnp.min(jnp.array([upper1,upper2]))
		ulower = jnp.max(jnp.array([lower1,lower2]))	
		
		uupper = jnp.max(jnp.array([upper1,upper2]))
		llower = jnp.min(jnp.array([lower1,lower2]))   
		
		poverlap = posi1[0]-posi2[0]
		up_dist = upper1-upper2
		lo_dist = lower1-lower2
		
		
		def small():
			return jnp.linspace(lower1,upper1,30)
		def nsmall():
			return jnp.linspace(lower2,upper2,30)
			
		test1 = jax.lax.cond(upper1-lower1<upper2-lower2,small,nsmall)
		test1 = jnp.concatenate((test1,jnp.array([lower1-2,upper1+2])))
		test1 = jnp.concatenate((test1,jnp.array([lower2-2,upper2+2])))
		
		overlap = 0
		def rloop(overlap,ind):
			mta = lip_sm(test1[ind],4,lower1,upper1)
			mtb = lip_sm(test1[ind],4,lower2,upper2)
			mtbL = jnp.floor(mtb)
			mtaI = jnp.array(mta,dtype=int)
			partb = mtb-mtbL
			mtbU = jnp.ceil(mtb)
				
			mtaL = jnp.floor(mta)
			mtbI = jnp.array(mtb,dtype=int)
			parta = mta-mtaL
			mtaU = jnp.ceil(mta)
			
			intLL = int_mat[jnp.array(mtaL,dtype=int),jnp.array(mtbL,dtype=int)]
			intLU = int_mat[jnp.array(mtaL,dtype=int),jnp.array(mtbU,dtype=int)]   
			intUL = int_mat[jnp.array(mtaU,dtype=int),jnp.array(mtbL,dtype=int)] 
			intUU = int_mat[jnp.array(mtaU,dtype=int),jnp.array(mtbU,dtype=int)] 
			
			intLLS = int_mat[jnp.array(mtbL,dtype=int),jnp.array(mtbL,dtype=int)]
			intLUS = int_mat[jnp.array(mtbL,dtype=int),jnp.array(mtbU,dtype=int)]   
			intULS = int_mat[jnp.array(mtbU,dtype=int),jnp.array(mtbL,dtype=int)] 
			intUUS = int_mat[jnp.array(mtbU,dtype=int),jnp.array(mtbU,dtype=int)]	  
					
			intVL = intLL*(1-partb)+partb*intLU
			
			intVU = intUL*(1-partb)+partb*intUU
			
			intV = intVL*(1-parta)+parta*intVU
			
			intVLS = intLLS*(1-partb)+partb*intLUS
			
			intVUS = intULS*(1-partb)+partb*intUUS
			
			intVS = intVLS*(1-parta)+parta*intVUS
			
			overlap += intV

			return overlap,ind
		overlap,_ = jax.lax.scan(rloop,overlap,jnp.arange(test1.shape[0]))

		return overlap,poverlap,up_dist,lo_dist
	def nclose():
		return 0.0,0.0,0.0,0.0
	overlap,poverlap,up_dist,lo_dist = jax.lax.cond(dist<100,close,nclose)
	flat_d = 10
	usplit = sj(up_dist-flat_d,1)*jnp.power(up_dist,2)+(1-sj(up_dist+flat_d,1))*jnp.power(up_dist,2)
	lsplit = sj(lo_dist-flat_d,1)*jnp.power(lo_dist,2)+(1-sj(lo_dist+flat_d,1))*jnp.power(lo_dist,2)
	return k1*overlap*-lj(dist,1.0)+lsplit+usplit

#calculates the full potential due to membrane-membrane interactions
@jax.jit
def nbh_calc(new_zs,k,k2,zd):
	new_zs = get_new_zs(new_zs,loop_inds)
	grad_grid = jnp.zeros((grid_size_x,grid_size_y,2))
	def nbh_grad_calc1(grad_grid,ind2):
		def nbh_grad_calc2(grad_grid,ind3):
			inder = flat_ind(ind2,ind3,grid_size_x)
			all_nbhs = jnp.array(get_nbh_inds(inder,grid_size_x,grid_size_y,1),dtype=int)
			tot = jnp.zeros(2)
			posi1 = new_zs[ind2,ind3]
			upper1 = zd+posi1[0]+0.5*posi1[1]
			lower1 = zd+posi1[0]-0.5*posi1[1]
			u_mult1 = bgrid_weight(ind2,ind3,upper1)
			l_mult1 = bgrid_weight(ind2,ind3,lower1)
			min_mult = (jnp.abs(u_mult1-l_mult1)+u_mult1+l_mult1)/2

			tot = tot.at[1].set(tot[1]+min_mult*0.5*k2*(mem_data[1]-new_zs[ind2,ind3,1])*(mem_data[1]-new_zs[ind2,ind3,1]))
			 			
			def nbh_grad_calc3(tot,ind):
				inda,indb = unflat_ind(all_nbhs[ind],grid_size_x)
				def diff(tot):						
					tot = tot.at[0].set(tot[0]+rpot_col(new_zs,inda,indb,ind2,ind3,0,zd,k))	
					return tot
				def ndiff(tot):
					return tot
				tot = jax.lax.cond(jnp.logical_and(inda==ind2,indb==ind3),ndiff,diff,tot)
				return tot,ind
			tot,_ = jax.lax.scan(nbh_grad_calc3,tot,jnp.arange(all_nbhs.shape[0]))
			grad_grid = grad_grid.at[ind2,ind3].set(tot)
			return grad_grid,ind3
		grad_grid,_ = jax.lax.scan(nbh_grad_calc2,grad_grid,jnp.arange(grid_size_y))
		return grad_grid,ind2
	grad_grid,_ = jax.lax.scan(nbh_grad_calc1,grad_grid,jnp.arange(grid_size_x))
	return grad_grid
	

#calculates the potential over all membrane segments
@jax.jit
def calc_graphs(xy,zd):
	xy = get_new_zs(xy,loop_inds)
	grid_vals = jnp.zeros((grid_size_x,grid_size_y,2))
	def calc_grid(grid_vals,ind):
		ind2 = ind
		def calc_grid2(grid_vals,ind):
			ind3 = ind
			z_disp = xy[ind2,ind3,0]+zd
			mt = xy[ind2,ind3,1]
			tot,_ = get_pot(ind2,ind3,xy[ind2,ind3,0],mt,zd)	
			grid_vals = grid_vals.at[ind2,ind3].set(tot)
			return grid_vals,ind
		grid_vals,_ = jax.lax.scan(calc_grid2,grid_vals,jnp.arange(grid_size_y))
		return grid_vals,ind
	grid_vals,_ = jax.lax.scan(calc_grid,grid_vals,jnp.arange(grid_size_x))
	return grid_vals
	
#used for evaluating potential over all membrane segments for creating Potential graphs
def get_pot_graph(in_xy):
	k = float(args.k1_const)
	k2 = float(args.k2_const)
	xy = in_xy[:grid_size_x*grid_size_y*2]
	zd = in_xy[-1]/5
	xy = xy.reshape((grid_size_x,grid_size_y,2))
	return calc_graphs(xy,zd)
	
#setting up the total potential function
@jax.jit
def total_pot_fun(in_xy):
	k = float(args.k1_const)
	k2 = float(args.k2_const)
	xy = in_xy[:grid_size_x*grid_size_y*2]
	zd = in_xy[-1]/5
	xy = xy.reshape((grid_size_x,grid_size_y,2))
	return jnp.sum(calc_graphs(xy,zd))+jnp.sum(nbh_calc(xy,k,k2,zd))
	
#differentiating using jax
ac_grad_fun = jax.grad(total_pot_fun,argnums=0)


#wrapping for use with numpy based minimisation algorithm (As the jax.numpy one had issues at time of writing - this could be updated once this works)
def nwrap_tpot(in_xy):
	in_xyj = jnp.array(in_xy)
	return total_pot_fun(in_xyj)
	
def nwrap_grad(in_xy):
	in_xyj = jnp.array(in_xy)
	test = np.array(ac_grad_fun(in_xyj))
	return test
	


new_zs = jnp.zeros((grid_size_x,grid_size_y,2))
new_zs = new_zs.at[:,:,1].set(mem_data[1])
start_z = float(args.zshift)*5
start_oz = 0


print("Misc..")
print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")
print("Minimisation... (This will take a while)")

timer = time.time()

#writing bead positions for debugging purposes
write_point(np.array(surf_poses),memdir+"Surface.pdb")
write_point(np.array(all_poses),memdir+"All_beads.pdb")

#main minimisation loop
try_again=True
while try_again:
	min_out = minimize(nwrap_tpot, np.concatenate((np.array(new_zs.flatten()),np.array([start_z]))),jac = nwrap_grad ,method='BFGS', tol=2e-4,options={"gtol":0.002,"maxiter":iters})
	new_zs = min_out.x[:grid_size_x*grid_size_y*2]
	start_z = min_out.x[-1]
	new_zs = new_zs.reshape((grid_size_x,grid_size_y,2))
	new_zs = np.array(get_new_zs(jnp.array(new_zs),loop_inds))
	total_z = np.mean(new_zs[loop_inds_end[:,0],loop_inds_end[:,1],0])
	new_zs[:,:,0] -= total_z
	start_z += total_z*5
	if(min_out.status == 2):
		try_again = False
	elif(min_out.status == 1):
		try_again = False
	else:
		try_again=not min_out.success


print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")

#calcuating graphs, contacts and general post processing
print("Post processing...")
timer = time.time()

np.savez(memdir+"Membrane_pos.npz",new_zs)
np.savez(memdir+"Z_change.npz",start_z/5.0)
np.savez(memdir+"Rotation_matrix.npz",rot_mat)


bfacs = calc_contacts(jnp.array(new_zs),jnp.array(all_poses),start_z/5.0,xs,ys)
bfacs_at = np.array(bfacs[1])
bfacs_hg = np.array(bfacs[0])

bfacs_at /= np.max(bfacs_at)
bfacs_hg /= np.max(bfacs_hg)

write_protein(bfacs_hg,fn,orient_dir+"HeadGroup_contacts.pdb")
write_protein(bfacs_at,fn,orient_dir+"AcylTail_contacts.pdb")


gc = calc_esf(start_z,new_zs,grid_ban,grid_size_x,grid_size_y,x_ext,y_ext,cposes,ctypes)
potg = get_pot_graph(min_out.x)
plt.imshow(potg[:,:,0])
plt.savefig(orient_dir+"Potential_Upper.png")
plt.clf()
plt.imshow(potg[:,:,1])
plt.savefig(orient_dir+"Potential_Lower.png")
plt.clf()


BotBan, TopBan = write_pdb(start_z,new_zs,grid_ban,grid_size_x,grid_size_y,x_ext,y_ext,gc,orient_dir,locdir,memdir,args.write_bfactors,mem_data,rot_mat_inv)

print("Z change:",np.format_float_positional(start_z/5.0,precision=3))
plt.imshow((new_zs[:,:,0]+0.5*new_zs[:,:,1])*TopBan)
plt.savefig(orient_dir+"Deformations_Upper.png")
plt.clf
plt.imshow((new_zs[:,:,0]-0.5*new_zs[:,:,1])*BotBan)
plt.savefig(orient_dir+"Deformations_Lower.png")
plt.clf



print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")



plt.imshow(gc[:,:,0]*BotBan)
plt.savefig(orient_dir+"Charge_Lower.png")
plt.clf
plt.imshow(gc[:,:,1]*TopBan)
plt.savefig(orient_dir+"Charge_Upper.png")
plt.clf


run_txt = open(orient_dir+"Run_command.txt","w")
run_txt.write(" ".join(sys.argv))
run_txt.close()

print("Total time:",str(np.format_float_positional(time.time()-tot_time,precision=3))+"s")

