import numpy 
from sysmor.mimoph2_coord import *
from sysmor.demos import build_iss
from itertools import product
from sysmor.mimoph2 import Weight 

def test_inner_loop(M = 1000, r = 50):
	np.random.seed(0)
	H = build_iss()
	p, m = H.output_dim, H.input_dim
	
	zs = {(i,j): [] for i, j in product(range(H.output_dim), range(H.input_dim))} 
	ys = {(i,j): [] for i, j in product(range(H.output_dim), range(H.input_dim))} 

	for k in range(M):
		i = np.random.randint(H.output_dim)	
		j = np.random.randint(H.input_dim)	
		z = np.random.uniform(0.1,1) + 1j*np.random.uniform(-100, 100)
		zs[i,j].append(complex(z))
		ys[i,j].append(complex(H.transfer(z, left_tangent = np.eye(p)[i], right_tangent = np.eye(m)[j])))
	
	for i,j in zs:
		print(i,j, len(zs[i,j]))

	#inner_loop
	weights = {}
	for i, j in zs:
		W = Weight(np.array(zs[i,j]))
		weights[i,j] = W @ np.eye(len(zs[i,j]))

	inner_loop(zs, ys, weights, r)	

if __name__ == '__main__':
	test_inner_loop()
