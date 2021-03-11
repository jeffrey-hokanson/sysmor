import numpy as np

from sysmor.tfirka2 import *
import sysmor, sysmor.demos

def test_tangential_interpolant():
	np.random.seed(0)
	H = sysmor.demos.build_iss()
	r = 8
	z = 1j*np.random.randn(r) + 1
	
	b = np.random.randn(r, H.input_dim) + 1j*np.random.randn(r, H.input_dim)
	c = np.random.randn(r, H.output_dim) + 1j*np.random.randn(r, H.output_dim)
	Hb = []
	cH = []
	cHpb = []

	for i in range(r):
		Hb.append( H.transfer(z[i], right_tangent = b[i]))
		cH.append( H.transfer(z[i], left_tangent = c[i]))
		cHpb.append( H.transfer(z[i], left_tangent = c[i], right_tangent = b[i], der = True)[1])
	
	# Build the interpolatory ROM
	H2 = tangential_hermite_interpolant(z, Hb, cH, cHpb, b, c)

	# Check interpolation conditions
	for i in range(r):
		Hb = H.transfer(z[i], right_tangent = b[i])
		Hb2 = H2.transfer(z[i], right_tangent = b[i])
		print('z', z[i])
		print("value right")
		print(Hb-Hb2)
		assert np.allclose(Hb, Hb2)
		
		cH = H.transfer(z[i], left_tangent = c[i])
		cH2 = H2.transfer(z[i], left_tangent = c[i])
		print("value left")
		print(cH - cH2)
		assert np.allclose(cH, cH2)
		
		_, cHpb = H.transfer(z[i], left_tangent = c[i], right_tangent = b[i], der = True)
		_, cHpb2 = H2.transfer(z[i], left_tangent = c[i], right_tangent = b[i], der = True)
		print("derivative")
		print(cHpb - cHpb2)
		assert np.allclose(cHpb, cHpb2)

def test_tfirka():
	np.random.seed(0)
	H = sysmor.demos.build_iss()
	r = 20
	
	z = 1j*np.random.randn(r) + 1e-1
	b = np.random.randn(r, H.input_dim) + 1j*np.random.randn(r, H.input_dim)
	c = np.random.randn(r, H.output_dim) + 1j*np.random.randn(r, H.output_dim)
	Hb = []
	cH = []
	cHpb = []

	for i in range(r):
		Hb.append( H.transfer(z[i], right_tangent = b[i]))
		cH.append( H.transfer(z[i], left_tangent = c[i]))
		cHpb.append( H.transfer(z[i], left_tangent = c[i], right_tangent = b[i], der = True)[1])

	
	#Hr = tangential_hermite_interpolant(z, Hb, cH, cHpb, b, c)

	tfirka(H, r, residue_correct = False)
	pass



if __name__ == '__main__':
	#test_tangential_interpolant()
	test_tfirka()
