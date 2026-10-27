import os
import sys
import warnings
import jax.numpy as jnp
from jax import tree_util
import jax
import jax.profiler
from enum import Enum
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import time
from mempro import MemPrO as ori
import argparse
from scipy.optimize import OptimizeResult
from scipy.optimize._optimize import _line_search_wolfe12, _LineSearchError


#BFGS as in scipy.optimize.minimize(method="BFGS") (scipy/optimize/_optimize.py, _minimize_bfgs; BSD licence),
#with the same line search and stopping rules. The only change is the inverse Hessian update: scipy forms
#A1 @ H @ A2 with two dense n x n matrix products (O(n^3), ~0.6 s per iteration for MemPrOD's ~3000 variables),
#whereas the identical update is applied here as a rank-2 correction (O(n^2)):
#  H+ = H - rho*(H y) s^T - rho*s (y^T H) + (rho^2 y^T H y + rho) s s^T
def bfgs_minimize(fun,x0,jac,gtol=1e-5,maxiter=None,c1=1e-4,c2=0.9,xrtol=0):
	x0 = np.asarray(x0,dtype=float).flatten()
	N = len(x0)
	if maxiter is None:
		maxiter = N*200
	cache = {}
	def cached(fn,key):
		def wrapped(x):
			if key not in cache or not np.array_equal(cache[key][0],x):
				cache[key] = (np.array(x,copy=True),fn(x))
			return cache[key][1]
		return wrapped
	f = cached(fun,"f")
	myfprime = cached(jac,"g")
	old_fval = f(x0)
	gfk = myfprime(x0)
	k = 0
	Hk = np.eye(N)
	old_old_fval = old_fval+np.linalg.norm(gfk)/2
	xk = x0
	warnflag = 0
	gnorm = np.max(np.abs(gfk))
	while (gnorm > gtol) and (k < maxiter):
		pk = -np.dot(Hk,gfk)
		try:
			alpha_k,fc,gc,old_fval,old_old_fval,gfkp1 = _line_search_wolfe12(f,myfprime,xk,pk,gfk,old_fval,old_old_fval,amin=1e-100,amax=1e100,c1=c1,c2=c2)
		except _LineSearchError:
			warnflag = 2
			break
		sk = alpha_k*pk
		xk = xk+sk
		if gfkp1 is None:
			gfkp1 = myfprime(xk)
		yk = gfkp1-gfk
		gfk = gfkp1
		k += 1
		gnorm = np.max(np.abs(gfk))
		if gnorm <= gtol:
			break
		if alpha_k*np.linalg.norm(pk) <= xrtol*(xrtol+np.linalg.norm(xk)):
			break
		if not np.isfinite(old_fval):
			warnflag = 2
			break
		rhok_inv = np.dot(yk,sk)
		rhok = 1000.0 if rhok_inv == 0. else 1./rhok_inv
		Hy = np.dot(Hk,yk)
		yH = np.dot(yk,Hk)
		cs = (rhok*rhok*np.dot(yk,Hy)+rhok)*sk-rhok*yH
		Hk += np.dot(np.stack((sk,-rhok*Hy),axis=1),np.stack((cs,sk)))
	if warnflag != 2 and k >= maxiter:
		warnflag = 1
	elif warnflag != 2 and (np.isnan(gnorm) or np.isnan(old_fval) or np.isnan(xk).any()):
		warnflag = 3
	return OptimizeResult(fun=old_fval,jac=gfk,hess_inv=Hk,status=warnflag,success=(warnflag == 0),x=xk,nit=k)


def main():
	warnings.filterwarnings('ignore')
	os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
	jax.config.update("jax_enable_x64",True)
	
	#PATH_TO_MARTINI overrides the Martini 3 force field bundled with MemPrO
	martini_default_path = os.environ.get("PATH_TO_MARTINI",os.path.join(os.path.dirname(ori.__file__),"martini_v3.itp"))
	
	#Reading command line args
	parser = argparse.ArgumentParser(prog="MemPrOD",description="This is a program for the prediction of deformations caused by membrane proteins.")
	parser.add_argument("-f", "--file_name",help = "Input file name (.pdb)")
	parser.add_argument("-o","--output",help="Name of the output directory (Default: Deformations)")
	parser.add_argument("-ni","--iters",help="Number of minimisation iterations (Default: 300)")
	parser.set_defaults(iters=300)
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
	parser.add_argument("-cut","--cutoff",help="Cutoff for LJ interactions (Default: 12)")
	parser.set_defaults(cutoff=12)
	parser.add_argument("-k1","--k1_const",help="Spring constant for neighbouring grid point Z deviations (Default: 0.25)")
	parser.set_defaults(k1_const=0.25)
	parser.add_argument("-k2","--k2_const",help="Spring constant for membrane compression (Default: 0)")
	parser.set_defaults(k2_const=0)
	parser.add_argument("-sm","--smooth",help="Width (Angstrom, Gaussian sigma) the upper and lower membrane surfaces are smoothed over after minimisation, each averaged over its own membrane cells, removing steps between neighbouring cells; 0 for none (Default: 5)")
	parser.set_defaults(smooth=5)
	parser.add_argument("-zshift","--zshift",help="Z shift of membrane for proteins that are misoriented due to lack of deformations.")
	parser.set_defaults(zshift=0)
	parser.add_argument("-mt","--membrane_thickness",help="Initial thickness of membrane in Angstroms (Default: 38)")
	parser.set_defaults(membrane_thickness=38.0)
	parser.add_argument("-wb","--write_bfactors",action="store_true",help="Toggle writing charge at each membrane segment instead of z position.")
	parser.set_defaults(write_bfactors=False)
	parser.add_argument("-enc","--enclosed",help="Membrane in regions the protein encloses (the inside of a ring or barrel): auto (where the narrowest core slice has at least 1 nm^2 free and at least 60%% of the lining residues are hydrophobic, as Insane4MemPrO -enclosed auto), lipids or empty (Default: auto)")
	parser.set_defaults(enclosed="auto")
	parser.add_argument("-res","--additional_residues",help="Comma separated list of additional residues in input files (eg: POPG) Note that only ATOM entries will be read, all HETATM entries will be ignored.")
	parser.set_defaults(additional_residues="")
	parser.add_argument("-res_itp","--additional_residues_itp_file",help="Path to the itp file describing all additional residues, and bead types associated to beads in the residue. A CG representation is required, for atomistic inputs an additional file is required which describes the beading.")
	parser.set_defaults(additional_residues_itp_file="")
	parser.add_argument("-res_cg","--residue_cg_file",help="Folder that contains files named RES.pdb describing the beading for each additional RES - see CG2AT github for examples")
	parser.set_defaults(residue_cg_file="")
	args = parser.parse_args()
	
	#Error checking user inputs
	add_reses = args.additional_residues.split(",")
	if(len(add_reses) > 0):
		if(len(add_reses[0]) > 0):
			print("Using additional residues:",", ".join(add_reses))
			if args.additional_residues_itp_file == "":
				print("ERROR: Additional residue itp is required if using additional residues.")
				exit()
			if args.residue_cg_file == "":
				print("WARNING: You have not added beading information for the added residues, this will cause an error if using an atomistic input.")
				
	if(len(add_reses) > 0):
		if(len(add_reses[0]) > 0):
			for i in add_reses:
				ori.add_Reses(i,args.additional_residues_itp_file)
				if len(args.residue_cg_file) > 0 :
					ori.add_AtomToBeads(args.residue_cg_file.lstrip("/")+"/"+i+".pdb")
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
		smooth = float(args.smooth)
	except:
		print("ERROR: Could not read value of -sm. Must be a float >= 0.")
		exit()
	if(smooth < 0):
		print("ERROR: Could not read value of -sm. Must be a float >= 0.")
		exit()
		
	if(args.enclosed not in ["auto","lipids","empty"]):
		print("ERROR: Could not read value of -enc. Must be auto, lipids or empty.")
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
			if len(c) > 54 and "[" not in c and c.startswith("ATOM"):			
				bbp = np.format_float_positional(bfacs[map_to_beads[bc]],precision=3)
				bbp += "0"*(3-len((bbp.split(".")[1])))
				new_c = c[:60]+(" "*(8-len(bbp)))+bbp+c[68:]
				bc+=1
				out_file.write(new_c)
			else:
				out_file.write(c)
		out_file.close()
		
		
	#lennard jones function with values <r cutoff	
	@jax.jit
	def lj(r,min_pos):
		r = jnp.maximum(r,min_pos*twopowsixth)
		return 4*(jnp.power((min_pos/r),12)-jnp.power((min_pos/r),6))
		
	
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
	
	#Splits loop_inds into consecutive groups whose updates are independent of each other
	#(no update reads or writes a cell written earlier in the same group), so each group can be
	#applied as one vector update while giving the same result as updating one index at a time
	def get_loop_groups(loop_inds):
		inds = np.array(loop_inds,dtype=int)
		groups = []
		start = 0
		written = set()
		for n,(ix,iy,dx,dy) in enumerate(inds):
			target = (ix,iy)
			source = (ix+dx,iy+dy)
			if target in written or source in written:
				groups.append(inds[start:n])
				start = n
				written = set()
			written.add(target)
		groups.append(inds[start:])
		return groups
		
	def get_new_zs(new_zs,loop_inds):
		for g in get_loop_groups(loop_inds):
			new_zs = new_zs.at[g[:,0],g[:,1],0].add(new_zs[g[:,0]+g[:,2],g[:,1]+g[:,3],0])
		return new_zs
		
		
		
		
			
		
	#Sigmoid function
	@jax.jit
	def sj(x,grad):
		ret_val = jnp.clip(-x*grad,-100.0,100.0) #Avoids overflow in exp
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
		#Same lookup as MemPrO's, which reads the ~16 MB force field once instead of once per bead pair
		return ori.get_int_strength(bead_1,bead_2,martini_file)
		
	#Gets the interations between protein beads and the enviroment. This is used to create the mean fields		
	def get_mem_def(martini_file):
		Beadtype_names = list(Beadtype.keys())
		no_beadtypes = len(Beadtype_names)
		#We use interactions strengths from martini using a POPE(Q4p)/POPG(P4)/POPC(Q1)? lipid as a template
		W_B_mins = jnp.array([get_int_strength("W",Beadtype_names[i],martini_file)-get_int_strength("W",Beadtype_names[i],martini_file) for i in range(no_beadtypes)])
		LH1_B_mins = jnp.array([get_int_strength("P4",Beadtype_names[i],martini_file)-get_int_strength("W",Beadtype_names[i],martini_file) for i in range(no_beadtypes)])
		LH2_B_mins = jnp.array([get_int_strength("Q5",Beadtype_names[i],martini_file)-get_int_strength("W",Beadtype_names[i],martini_file) for i in range(no_beadtypes)])
		LH3_B_mins = jnp.array([get_int_strength("SN4a",Beadtype_names[i],martini_file)-get_int_strength("W",Beadtype_names[i],martini_file) for i in range(no_beadtypes)])
		LH4_B_mins = jnp.array([get_int_strength("N4a",Beadtype_names[i],martini_file)-get_int_strength("W",Beadtype_names[i],martini_file) for i in range(no_beadtypes)])
		LT1_B_mins = jnp.array([get_int_strength("C1",Beadtype_names[i],martini_file)-get_int_strength("W",Beadtype_names[i],martini_file) for i in range(no_beadtypes)])
		LT2_B_mins = jnp.array([get_int_strength("C4h",Beadtype_names[i],martini_file)-get_int_strength("W",Beadtype_names[i],martini_file) for i in range(no_beadtypes)])
		Charge_B_mins =jnp.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype="float64")
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
		
	#Gaussian smoothing of the upper (z+t/2) and lower (z-t/2) surfaces of new_zs (midplane z, thickness t), each over
	#the cells where its leaflet has membrane (normalised convolution), so the protein does not drag the edge down
	def smooth_surfaces(new_zs,top_ban,bot_ban,sigx,sigy):
		from scipy import ndimage
		new_zs = np.array(new_zs)
		surfs = []
		for sgn,ban in ((1,top_ban),(-1,bot_ban)):
			surf = new_zs[:,:,0]+sgn*0.5*new_zs[:,:,1]
			w = (np.array(ban) > 0.5).astype(float)
			num = ndimage.gaussian_filter(surf*w,(sigx,sigy),mode="nearest")
			den = ndimage.gaussian_filter(w,(sigx,sigy),mode="nearest")
			surfs.append(np.where(den > 1e-3,num/np.maximum(den,1e-3),surf))
		new_zs[:,:,0] = 0.5*(surfs[0]+surfs[1])
		new_zs[:,:,1] = surfs[0]-surfs[1]
		return new_zs
		
	#calcualtes the eleectrostatic force using below
	#All lower/upper head positions of all grid cells are evaluated together; cells outside the membrane stay 0
	def calc_esf(start_z,grid_in,grid_ban,nx,ny,exx,exy,cposes,ctypes):
		grid_in = jnp.array(grid_in)
		xs = np.linspace(-exx,exx,nx)
		ys = np.linspace(-exy,exy,ny)
		ms = jax.vmap(jax.vmap(get_mem_struc))(grid_in[:,:,1])
		z_pos = jnp.stack((grid_in[:,:,0]+(ms[:,:,0]+start_z/5),grid_in[:,:,0]+(ms[:,:,7]+start_z/5)),axis=2)
		x_pos,y_pos = np.meshgrid(xs,ys,indexing="ij")
		pos = jnp.stack((jnp.broadcast_to(x_pos[:,:,None],z_pos.shape),jnp.broadcast_to(y_pos[:,:,None],z_pos.shape),z_pos),axis=3).reshape(-1,3)
		charge = np.array(jax.lax.map(lambda p: calc_esf_p(p,cposes,ctypes),pos,batch_size=256)).reshape(nx,ny,2)
		return np.where(np.array(grid_ban)[:,:,None] == 1,charge,0.0)
			
	#calcues the electrostatic force
	@jax.jit			
	def calc_esf_p(pos,cposes,ctypes):
		ctypes = jnp.array(ctypes,dtype=int)
		charge_const = 92.64
		Charge_B_mins =jnp.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype="float64")
		return jnp.sum(Charge_B_mins[ctypes]*charge_const/jnp.linalg.norm(pos-cposes,axis=1))
							
						
	
		
			
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
	
	Beadtype = ori.Beadtype
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
	
	if(surf_poses.shape[0] == 0):
		print("ERROR: No protein surface within 50 A of the membrane midplane (z = 0). The input must be oriented in the membrane, for example a MemPrO output structure.")
		exit()
	
	indexer2 = np.nonzero((np.array(poses[:,2]) < 50)*(np.array(poses[:,2] > -50)))
	
	all_poses = poses.copy()
	poses = poses[indexer2]
	
	cutoff = int(args.cutoff)
	
	x_ext = max(-jnp.min(surf_poses[:,0]),jnp.max(surf_poses[:,0]))+25
	y_ext = max(-jnp.min(surf_poses[:,1]),jnp.max(surf_poses[:,1]))+25
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
	#Slot k of a grid cell holds the k-th bead (in index order) that falls in that cell; the last slot is the count
	def allo_bead(surf_poses):
		no_beads = surf_poses.shape[0]
		xgrid,ygrid = put_into_grid([surf_poses[:,0],surf_poses[:,1]],[2*x_ext,2*y_ext],[nxx,nyy],[-x_ext,-y_ext])
		xgrid = np.array(xgrid)
		ygrid = np.array(ygrid)
		big_grid = np.zeros((grid_size_x,grid_size_y,no_beads+1),dtype=int)-1
		big_grid[:,:,no_beads] = 0
		for bind in range(no_beads):
			big_grid[xgrid[bind],ygrid[bind],big_grid[xgrid[bind],ygrid[bind],-1]] = bind
			big_grid[xgrid[bind],ygrid[bind],-1] += 1
		return jnp.array(big_grid)
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
		max_beads = int(jnp.max(big_grid[:,:,-1]))
		sq = jnp.array([[0,0],[-1,0],[1,0],[0,-1],[0,1],[-1,1],[-1,-1],[1,1],[1,-1]],dtype=int)
		@partial(jax.vmap,in_axes=0)
		def agrid_vmap(inds):
			ind2 = inds[0]
			ind3 = inds[1]
			indx,indy = put_into_grid([xs[ind2],ys[ind3]],[2*x_ext,2*y_ext],[nxx,nyy],[-x_ext,-y_ext])
			#candidate (slot,neighbour) pairs in the order the original nested loops visited them
			bginds = big_grid[(indx+sq[:,0])%grid_size_x,(indy+sq[:,1])%grid_size_y][:,:max_beads].T.reshape(-1)
			dists = jnp.linalg.norm(surf_poses[bginds][:,0:2]-jnp.array([xs[ind2],ys[ind3]]),axis=1)
			keep = (bginds != -1)*(dists < cutoff_val)
			slot = jnp.where(keep,jnp.cumsum(keep)-1,no_beads+1)
			grid_p = jnp.zeros((no_beads+1,2),dtype=float)-1
			grid_p = grid_p.at[slot,0].set(bginds,mode="drop").at[slot,1].set(dists,mode="drop")
			grid_p = grid_p.at[no_beads,0].set(jnp.sum(keep)).at[no_beads,1].set(0)
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
	
	#Every blocking bead clears the vertical cells indap-1, indap and indap+1 of its grid cell (order does not matter)
	block_binds = jnp.array(grid_block[:,:,:max_beads_block,0],dtype=int)
	block_ix,block_iy = np.meshgrid(np.arange(grid_size_x),np.arange(grid_size_y),indexing="ij")
	block_ix = jnp.broadcast_to(block_ix[:,:,None],block_binds.shape)
	block_iy = jnp.broadcast_to(block_iy[:,:,None],block_binds.shape)
	indap = jnp.array(((poses[jnp.where(block_binds != -1,block_binds,0),2]+vext)/(2*vext))*vert_grid_size,dtype=int)
	for shift in [0,-1,1]:
		#indices are normalised and dropped by .at exactly as in the original per-bead updates; empty slots are dropped
		zi = jnp.where(block_binds != -1,indap+shift,vert_grid_size)
		bgrid = bgrid.at[block_ix,block_iy,zi].set(0,mode="drop")
	
				
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
	
	#Side-chain atoms (or beads) of the input with their residue names and residue numbers, in the grid frame
	def side_chains():
		backbone = ["N","CA","C","O","OXT","BB"]
		xyz,names,res = [],[],[]
		prev_key = None
		nres = -1
		for c in open(fn):
			if(c.startswith("ATOM") and len(c) > 54):
				key = (c[17:21].strip(),c[21],c[22:27])
				if(key != prev_key):
					nres += 1
					prev_key = key
				if(c[12:16].strip() in backbone or c[76:78].strip() == "H" or c[12:16].strip().startswith("H")):
					continue
				xyz.append([float(c[30:38]),float(c[38:46]),float(c[46:54])])
				names.append(key[0])
				res.append(nres)
		xyz = np.dot(np.array(rot_mat),np.array(xyz).T).T if len(xyz) > 0 else np.zeros((0,3))
		return xyz,np.array(names),np.array(res,dtype=int)
	
	def fill_enclosed(bgrid,bgrid_o,acc):
		from scipy import ndimage
		from scipy.spatial import cKDTree
		hydrophobic = ["ALA","VAL","LEU","ILE","MET","PHE","TRP","CYS","PRO"]
		zshift = float(args.zshift)
		zs = -vext+(np.arange(vert_grid_size)+0.5)*(2*vext/vert_grid_size)
		core = np.nonzero(np.abs(zs-zshift) < mem_data[1]/2.0)[0]
		#Slices from the midplane out to the head groups, where enclosed regions that qualify are filled
		mem_slices = np.nonzero(np.abs(zs-zshift) < mem_data[1]/2.0+10)[0]
		enclosed = (acc > 0.5)*(bgrid_o < 0.5)
		regions,nregions = ndimage.label(np.any(enclosed[:,:,core],axis=2))
		if(nregions == 0):
			return jnp.array(bgrid),jnp.array(bgrid_o)
		sc_xyz,sc_names,sc_res = side_chains()
		gx,gy = np.meshgrid(np.array(xs),np.array(ys),indexing="ij")
		pix_area = grid_den*grid_den/100.0
		near = 6.0
		print("Membrane in regions the protein encloses (-enc "+args.enclosed+"):")
		for reg in range(1,nregions+1):
			mask = regions == reg
			free = np.array([np.sum(acc[:,:,k][mask] > 0.5) for k in core])
			narrowest = np.min(free)*pix_area
			lining = set()
			for k in core:
				pix = mask*(acc[:,:,k] > 0.5)
				if(not np.any(pix)):
					continue
				in_slice = np.abs(sc_xyz[:,2]-zs[k]) < vext/vert_grid_size
				if(not np.any(in_slice)):
					continue
				d,_ = cKDTree(np.stack((gx[pix],gy[pix]),axis=1)).query(sc_xyz[in_slice,:2])
				lining.update(sc_res[in_slice][d < near].tolist())
			lining = sorted(lining)
			n_hyd = sum([sc_names[sc_res == r][0] in hydrophobic for r in lining])
			frac = n_hyd/len(lining) if len(lining) > 0 else 0.0
			centre = np.dot(np.array(rot_mat_inv),np.array([gx[mask].mean(),gy[mask].mean(),0]))
			fill = args.enclosed == "lipids" or (narrowest >= 1.0 and frac >= 0.6)
			print("  Region at ("+str(np.round(centre[0]/10,1))+", "+str(np.round(centre[1]/10,1))+") nm: narrowest core slice "+str(np.round(narrowest,1))+" nm^2 free, "+str(len(lining))+" lining residues, "+str(int(round(100*frac)))+"% hydrophobic: "+("membrane" if fill else "left empty"))
			if(not fill):
				continue
			for k in mem_slices:
				pix = mask*enclosed[:,:,k]
				if(not np.any(pix)):
					continue
				#As the flood fill marks the cells it reaches and their neighbours
				reach = ndimage.binary_dilation(pix,structure=np.ones((3,3)))
				bgrid_o[:,:,k] = np.maximum(bgrid_o[:,:,k],reach)
				bgrid[:,:,k] = np.maximum(bgrid[:,:,k],ndimage.binary_dilation(reach,structure=np.ones((3,3))))
		return jnp.array(bgrid),jnp.array(bgrid_o)
	
	#Each height slice is filled independently, so all slices are processed together
	def fill_out_slice(bslice):
		acc = accessable(bslice)
		tempb = fill_mid(1-acc,0.5)
		return fill_mid(1-tempb,0.5),tempb,acc
	bgrid,bgrid_o,bgrid_acc = jax.vmap(fill_out_slice,in_axes=2,out_axes=2)(bgrid)
	
	#Regions the protein encloses (room for lipids that cannot be reached from outside the protein) are treated as
	#protein above. With -enc auto they are membrane where the narrowest slice of the hydrophobic core has at least
	#1 nm^2 free and at least 60% of the residues lining it are hydrophobic, the rule of Insane4MemPrO -enclosed auto.
	if(args.enclosed != "empty"):
		bgrid,bgrid_o = fill_enclosed(np.array(bgrid),np.array(bgrid_o),np.array(bgrid_acc))
	
	print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")
	timer = time.time()
	
	#This function indicates if grid point (i,j) and height z is within the protein.
	@jax.jit
	def bgrid_weight(i,j,z):
		zind = ((z+vext)/(2*vext))*vert_grid_size
		zind_l = jnp.array(jnp.floor(zind),dtype=int)-5
		tot = jnp.sum(bgrid[i,j,jnp.clip(zind_l+jnp.arange(10),0,vert_grid_size-1)])
		zind_u = jnp.clip(zind_l+10,0,vert_grid_size-1)
		zind_l = jnp.clip(zind_l,0,vert_grid_size-1)
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
		sumer = jnp.sum(jax.vmap(bgrid_weight,in_axes=(None,None,0))(ind2,ind3,logs+z_disp))
		return (sj(-sumer+4,3))*(-sumer-200)/5.0
			
	
	
	#Potential function. If surface bead or membrane segment is within the protein it's contribution is ignored
	#Vectorised over bead slots: empty slots (index -1) and beads outside the cutoff are masked out
	def get_pot(ind2,ind3,zt,mt,zd):
		no_beads = grid[ind2,ind3,-1,0]
		z_disp = zt+zd
		mem_structure = get_mem_struc(mt)
		binds = jnp.array(grid[ind2,ind3,:int(max_beads),0],dtype=int)
		dists = grid[ind2,ind3,:int(max_beads),1]
		valid = binds > -1e-1
		safe_binds = jnp.where(valid,binds,0)
		z_pos = surf_poses[safe_binds,2]-z_disp
		indx,indy = put_into_grid([surf_poses[safe_binds,0],surf_poses[safe_binds,1]],[2*x_ext,2*y_ext],[nxx,nyy],[-x_ext,-y_ext])
		bead_num = bead_types[safe_binds]
		disterz = jnp.sqrt(dists*dists)
		contrib = (jax.vmap(smj,in_axes=(0,None,0,None))(z_pos,0.5,bead_num,mem_structure)*-1*jax.vmap(lj,in_axes=(0,None))(disterz,8.0)
			*jax.vmap(bgrid_weight,in_axes=(None,None,0))(ind2,ind3,z_pos)*jax.vmap(bgrid_weight,in_axes=(0,0,0))(indx,indy,z_pos))
		contrib = jnp.where(valid*(dists<float(args.cutoff))*(dists > 0),contrib,0.0)
		edge = jnp.logical_or(jnp.logical_or(ind3==0,ind3==grid.shape[1]-1),jnp.logical_or(ind2==0,ind2==grid.shape[0]-1))
		s_val = jnp.where(edge,1e-7,1e-6)*((zt*zt)+((mt-mem_data[1])*(mt-mem_data[1])))
		tot = jnp.array([s_val+jnp.sum(jnp.where(z_pos>0,contrib,0.0)),s_val+jnp.sum(jnp.where(z_pos>0,0.0,contrib))])
		tot += intbgrid(ind2,ind3,zt,mt,zd)
		return tot,no_beads
			
	#Calculates contacts with head groups and acyl tails		
	#Vectorised over protein beads and the 40 tail samples; grid cells are processed in batches to bound memory
	@jax.jit
	def calc_contacts(new_zs,poses,zdisp,xs,ys):
		mindd = 3
		cell_i,cell_j = np.meshgrid(np.arange(new_zs.shape[0]),np.arange(new_zs.shape[1]),indexing="ij")
		cell_i,cell_j = jnp.array(cell_i.flatten()),jnp.array(cell_j.flatten())
		def cell(ind):
			ind1,ind2 = cell_i[ind],cell_j[ind]
			mt = new_zs[ind1,ind2,1]
			z = new_zs[ind1,ind2,0]
			pos1 = jnp.array([xs[ind1],ys[ind2],z-mt/2.0-4+zdisp])
			pos2 = jnp.array([xs[ind1],ys[ind2],z+mt/2.0+4+zdisp])
			dist1 = jnp.linalg.norm(poses-pos1,axis=1)/10
			dist2 = jnp.linalg.norm(poses-pos2,axis=1)/10
			bfacH = jnp.exp(-jnp.maximum(dist1*dist1,mindd))*bgrid_weight(ind1,ind2,z-mt/2.0+4+zdisp)+jnp.exp(-jnp.maximum(dist2*dist2,mindd))*bgrid_weight(ind1,ind2,z-mt/2.0-4+zdisp)
			zscan = jnp.linspace(pos1[2],pos2[2],40)
			pos3 = jnp.stack((jnp.full(40,pos1[0]),jnp.full(40,pos1[1]),zscan),axis=1)
			dist3 = jnp.linalg.norm(poses[:,None,:]-pos3[None,:,:],axis=2)/10
			weights = jax.vmap(bgrid_weight,in_axes=(None,None,0))(ind1,ind2,zscan)
			bfacT = jnp.sum(jnp.exp(-jnp.maximum(dist3*dist3,mindd))*weights[None,:],axis=1)
			return bfacH,bfacT
		bfacsH,bfacsT = jax.lax.map(cell,jnp.arange(cell_i.shape[0]),batch_size=16)
		return jnp.sum(bfacsH,axis=0),jnp.sum(bfacsT,axis=0)
					
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
		posi1 = new_zs[inda1,indb1]
		upper1 = zd+posi1[0]+0.5*posi1[1]
		lower1 = zd+posi1[0]-0.5*posi1[1]
		
		posi2 = new_zs[inda2,indb2]
		upper2 = zd+posi2[0]+0.5*posi2[1]
		lower2 = zd+posi2[0]-0.5*posi2[1]
		
		poverlap = posi1[0]-posi2[0]
		up_dist = upper1-upper2
		lo_dist = lower1-lower2
		
		test1 = jnp.where(upper1-lower1<upper2-lower2,jnp.linspace(lower1,upper1,30),jnp.linspace(lower2,upper2,30))
		test1 = jnp.concatenate((test1,jnp.array([lower1-2,upper1+2])))
		test1 = jnp.concatenate((test1,jnp.array([lower2-2,upper2+2])))
		
		mta = lip_sm(test1,4,lower1,upper1)
		mtb = lip_sm(test1,4,lower2,upper2)
		mtbL = jnp.floor(mtb)
		partb = mtb-mtbL
		mtbU = jnp.ceil(mtb)
		mtaL = jnp.floor(mta)
		parta = mta-mtaL
		mtaU = jnp.ceil(mta)
		intLL = int_mat[jnp.array(mtaL,dtype=int),jnp.array(mtbL,dtype=int)]
		intLU = int_mat[jnp.array(mtaL,dtype=int),jnp.array(mtbU,dtype=int)]
		intUL = int_mat[jnp.array(mtaU,dtype=int),jnp.array(mtbL,dtype=int)]
		intUU = int_mat[jnp.array(mtaU,dtype=int),jnp.array(mtbU,dtype=int)]
		intVL = intLL*(1-partb)+partb*intLU
		intVU = intUL*(1-partb)+partb*intUU
		overlap = jnp.sum(intVL*(1-parta)+parta*intVU)
		
		#Segments further apart than 100 grid points do not interact
		close = dist<100
		overlap = jnp.where(close,overlap,0.0)
		up_dist = jnp.where(close,up_dist,0.0)
		lo_dist = jnp.where(close,lo_dist,0.0)
		flat_d = 10
		usplit = sj(up_dist-flat_d,1)*jnp.power(up_dist,2)+(1-sj(up_dist+flat_d,1))*jnp.power(up_dist,2)
		lsplit = sj(lo_dist-flat_d,1)*jnp.power(lo_dist,2)+(1-sj(lo_dist+flat_d,1))*jnp.power(lo_dist,2)
		return k1*overlap*-lj(dist,1.0)+lsplit+usplit
	
	#calculates the full potential due to membrane-membrane interactions
	#Neighbour table (the 3x3 block around each cell, wrapped at the edges as in get_nbh_inds) built once with numpy
	def get_nbh_table(nx,ny):
		table = np.zeros((nx,ny,9,2),dtype=int)
		for ix in range(nx):
			for iy in range(ny):
				for a,dx in enumerate([-1,0,1]):
					for b,dy in enumerate([-1,0,1]):
						new_x = ix+dx
						new_y = iy+dy
						new_x = new_x-nx if new_x >= nx else new_x
						new_y = new_y-ny if new_y >= ny else new_y
						new_x = new_x+nx if new_x < 0 else new_x
						new_y = new_y+ny if new_y < 0 else new_y
						table[ix,iy,a*3+b] = [new_x,new_y]
		return table
	nbh_table = jnp.array(get_nbh_table(grid_size_x,grid_size_y))
	cell_x,cell_y = np.meshgrid(np.arange(grid_size_x),np.arange(grid_size_y),indexing="ij")
	
	@jax.jit
	def nbh_calc(new_zs,k,k2,zd):
		new_zs = get_new_zs(new_zs,loop_inds)
		def cell(ind2,ind3):
			posi1 = new_zs[ind2,ind3]
			upper1 = zd+posi1[0]+0.5*posi1[1]
			lower1 = zd+posi1[0]-0.5*posi1[1]
			u_mult1 = bgrid_weight(ind2,ind3,upper1)
			l_mult1 = bgrid_weight(ind2,ind3,lower1)
			min_mult = (jnp.abs(u_mult1-l_mult1)+u_mult1+l_mult1)/2
			thick = min_mult*0.5*k2*(mem_data[1]-new_zs[ind2,ind3,1])*(mem_data[1]-new_zs[ind2,ind3,1])
			nbhs = nbh_table[ind2,ind3]
			pots = jax.vmap(rpot_col,in_axes=(None,0,0,None,None,None,None,None))(new_zs,nbhs[:,0],nbhs[:,1],ind2,ind3,0,zd,k)
			not_self = jnp.logical_not(jnp.logical_and(nbhs[:,0]==ind2,nbhs[:,1]==ind3))
			return jnp.array([jnp.sum(jnp.where(not_self,pots,0.0)),thick])
		return jax.vmap(jax.vmap(cell))(cell_x,cell_y)
		
	
	#calculates the potential over all membrane segments
	@jax.jit
	def calc_graphs(xy,zd):
		xy = get_new_zs(xy,loop_inds)
		def cell(ind2,ind3):
			tot,_ = get_pot(ind2,ind3,xy[ind2,ind3,0],xy[ind2,ind3,1],zd)
			return tot
		return jax.vmap(jax.vmap(cell))(cell_x,cell_y)
		
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
	ac_grad_fun = jax.jit(jax.grad(total_pot_fun,argnums=0))
	
	
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
		min_out = bfgs_minimize(nwrap_tpot, np.concatenate((np.array(new_zs.flatten()),np.array([start_z]))),jac = nwrap_grad,gtol=0.002,maxiter=iters)
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
	
	#Smoothing the upper and lower surfaces, each averaged over the cells where that leaflet has membrane
	if(smooth > 0):
		BotBan, TopBan = write_pdb(start_z,new_zs,grid_ban,grid_size_x,grid_size_y,x_ext,y_ext,None,orient_dir,locdir,memdir,False,mem_data,rot_mat_inv)
		new_zs = smooth_surfaces(new_zs,TopBan,BotBan,smooth/(2*x_ext/(grid_size_x-1)),smooth/(2*y_ext/(grid_size_y-1)))
	
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
	plt.savefig(orient_dir+"Potential_Upper.svg")
	plt.clf()
	plt.imshow(potg[:,:,1])
	plt.savefig(orient_dir+"Potential_Lower.svg")
	plt.clf()
	
	
	BotBan, TopBan = write_pdb(start_z,new_zs,grid_ban,grid_size_x,grid_size_y,x_ext,y_ext,gc,orient_dir,locdir,memdir,args.write_bfactors,mem_data,rot_mat_inv)
	
	print("Z change:",np.format_float_positional(start_z/5.0,precision=3))
	plt.imshow((new_zs[:,:,0]+0.5*new_zs[:,:,1])*TopBan)
	plt.savefig(orient_dir+"Deformations_Upper.svg")
	plt.clf()
	plt.imshow((new_zs[:,:,0]-0.5*new_zs[:,:,1])*BotBan)
	plt.savefig(orient_dir+"Deformations_Lower.svg")
	plt.clf()
	
	
	
	print("Done in:", str(np.format_float_positional(time.time()-timer,precision=3))+"s")
	
	
	
	plt.imshow(gc[:,:,0]*BotBan)
	plt.savefig(orient_dir+"Charge_Lower.svg")
	plt.clf()
	plt.imshow(gc[:,:,1]*TopBan)
	plt.savefig(orient_dir+"Charge_Upper.svg")
	plt.clf()
	
	
	run_txt = open(orient_dir+"Run_command.txt","w")
	run_txt.write(" ".join(sys.argv))
	run_txt.close()
	
	print("Total time:",str(np.format_float_positional(time.time()-tot_time,precision=3))+"s")


if __name__ == "__main__":
	main()