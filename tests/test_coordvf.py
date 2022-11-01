import numpy as np


from sysmor.coordvf import *
import sysmor, sysmor.demos


def test_coordinate_h2_vecfit():
	H = sysmor.demos.build_iss()
	r = 4

	z = 1e-3 + 1j*np.linspace(-100, 100, 20)

	m = H.input_dim
	p = H.output_dim
	zs = {(i,j): [] for i, j in product(range(p), range(m))}
	ys = {(i,j): [] for i, j in product(range(p), range(m))}

	for k in range(len(z)):
		for j in range(H.input_dim):
			for i in range(H.output_dim):
				zs[i,j].append(complex(z[k]))
				ys[i,j].append(complex(
					H.transfer(z[k], 
						left_tangent = np.eye(H.output_dim)[i],
						right_tangent = np.eye(H.input_dim)[j]
						)
					))

	coordinate_vecfit(zs, ys, r-1, r)
	

if __name__ == '__main__':
	test_coordinate_h2_vecfit()
