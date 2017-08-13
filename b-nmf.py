#-*- coding: utf-8 -*-
import sys
from numpy import *
from numpy.linalg import *

reload(sys)
sys.setdefaultencoding('utf-8')

def FirstNMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda):
	'Step 1. First NMF of X'
	#初始化W,Hh
	n,m = X.shape
	Hh = mat(random.rand(D,m))
	#print 'Hh',Hh.shape
	
	#construct L
	d = array(U).sum(axis=1)
	D = mat(diag(d.T))

	#update Hh
	AH = W.H * multiply(P,W*Hh) + beta * Hh + gamma * Hh + 1e-8
	BH = W.H * multiply(P,X) + gamma * H * S
	Hh = multiply(Hh,BH/AH)

	#Optimaize W and Hh
	i = 0
	while i<2000:
		#AW = multiply(P,W*Hh) * Hh.H + alpha * W + 2 * mlambda * W * W.H * W + 1e-8
		#BW = multiply(P,X) * Hh.H + 2 * mlambda * U * W
		AW = multiply(P,W*Hh) * Hh.H + alpha * W + mlambda * D * W + 1e-8
		BW = multiply(P,X) * Hh.H + mlambda * U * W
		W = multiply(W,BW/AW)

		AH = W.H * multiply(P,W*Hh) + beta * Hh + gamma * Hh + 1e-8
		BH = W.H * multiply(P,X) + gamma * H * S
		Hh = multiply(Hh,BH/AH)

		i += 1

	#print W
	#print Hh	
	return W,Hh

def NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda):
	'Step 2. NMF of Y'
	#初始化Wm,H
	Wm = R * W
	#print 'Wm',Wm.shape
	
	#construct L
	d = array(V).sum(axis=1)
	D = mat(diag(d.T))

	#update H
	#AH = Wm.H * multiply(Q,Wm*H) + beta * H + 2 * mlambda * H * H.H * H + 1e-8
	#BH = Wm.H * multiply(Q,Y) + 2 * mlambda * H * V
	AH = Wm.H * multiply(Q,Wm*H) + beta * H + mlambda * H * D + 1e-8
	BH = Wm.H * multiply(Q,Y) + mlambda * H * V
	H = multiply(H,BH/AH)

	#Optimatize Wm and H
	i = 0
	while i<2000:
		AW = multiply(Q,Wm*H) * H.H + alpha * Wm + gamma * Wm + 1e-8
		BW = multiply(Q,Y) * H.H + gamma * R * W
		Wm = multiply(Wm,BW/AW)

		#AH = Wm.H * multiply(Q,Wm*H) + beta * H + 2 * mlambda * H * H.H * H + 1e-8
		#BH = Wm.H * multiply(Q,Y) + 2 * mlambda * H * V
		AH = Wm.H * multiply(Q,Wm*H) + beta * H + mlambda * H * D + 1e-8
		BH = Wm.H * multiply(Q,Y) + mlambda * H * V
		H = multiply(H,BH/AH)

		i += 1

	#print Wm
	#print H
	return Wm,H

def NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda):
	'Step 3. Subsequent NMF of X'
	#初始化W,Hh
	Hh = H * S
	
	#construct L
	d = array(U).sum(axis=1)
	D = mat(diag(d.T))

	#update W
	#AW = multiply(P,W*Hh) * Hh.H + alpha * W + 2 * mlambda * W * W.H * W + 1e-8
	#BW = multiply(P,X) * Hh.H + 2 * mlambda * U * W
	AW = multiply(P,W*Hh) * Hh.H + alpha * W + mlambda * D * W + 1e-8
	BW = multiply(P,X) * Hh.H + mlambda * U * W
	W = multiply(W,BW/AW)

	#Optimaize W and Hh
	i = 0
	while i<2000:
		AH = W.H * multiply(P,W*Hh) + beta * Hh + gamma * Hh + 1e-8
		BH = W.H * multiply(P,X) + gamma * H * S
		Hh = multiply(Hh,BH/AH)

		#AW = multiply(P,W*Hh) * Hh.H + alpha * W + 2 * mlambda * W * W.H * W + 1e-8
		#BW = multiply(P,X) * Hh.H + 2 * mlambda * U * W
		AW = multiply(P,W*Hh) * Hh.H + alpha * W + mlambda * D * W + 1e-8
		BW = multiply(P,X) * Hh.H + mlambda * U * W
		W = multiply(W,BW/AW)

		i += 1

	#print W
	#print Hh	
	return W,Hh

def B_NMF_Z1(G,E,U,V,x,y,alpha,beta,gamma,mlambda) :
	'B_NMF解决稀疏矩阵问题'
	#将G矩阵分解为X,Y
	X = G[:,:x]
	Y = G[y:,:]
	#将E矩阵分解为P,Q
	P = E[:,:x]
	Q = E[y:,:]
	nx,mx = X.shape
	ny,my = Y.shape
	#D = int(0.1 * min(x,y*9))
	D = 5
	W = mat(random.rand(nx,D))
	H = mat(random.rand(D,my))

	R = mat(ones((ny,ny)))
	R = diag(diag(R))
	R1 = mat(zeros((ny,nx-ny)))
	R = hstack((R1,R))

	S = mat(ones((mx,mx)))
	S = diag(diag(S))
	S1 = mat(zeros((my-mx,mx)))
	S = vstack((S,S1))

	#Step 1. First NMF of X
	W,Hh = FirstNMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)
	print "Step 1. Over..."

	#Step 2. NMF of Y
	Wm,H = NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda)
	print "Step 2. Over..."

	#Step 3. Subsequent NMF of X
	W,Hh = NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)
	print "Step 3. Over..."

	#Step 4. Repeat Steps 2 and 3.
	i = 0
	while i<10:
		Wm,H = NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda)
		W,Hh = NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)

		i += 1
		print i
	print "Step 4. Over..."

	#Z = WH
	Z = W * H
	#filename = r"..\data\Z1.txt"
	#savetxt(filename, around(Z, decimals=2), fmt="%.2f")
	return Z

def B_NMF_Z2(G,E,U,V,x,y,alpha,beta,gamma,mlambda) :
	'B_NMF解决稀疏矩阵问题'
	#将G矩阵分解为X,Y
	X = G[:,x:]
	Y = G[y:,:]
	#将E矩阵分解为P,Q
	P = E[:,x:]
	Q = E[y:,:]
	nx,mx = X.shape
	ny,my = Y.shape
	#D = int(0.9 * min(x,y))
	D = 5
	W = mat(random.rand(nx,D))
	H = mat(random.rand(D,my))

	R = mat(ones((ny,ny)))
	R = diag(diag(R))
	R1 = mat(zeros((ny,nx-ny)))
	R = hstack((R1,R))

	S = mat(ones((mx,mx)))
	S = diag(diag(S))
	S1 = mat(zeros((my-mx,mx)))
	S = vstack((S1,S))

	#Step 1. First NMF of X
	W,Hh = FirstNMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)
	print "Step 1. Over..."

	#Step 2. NMF of Y
	Wm,H = NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda)
	print "Step 2. Over..."

	#Step 3. Subsequent NMF of X
	W,Hh = NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)
	print "Step 3. Over..."

	#Step 4. Repeat Steps 2 and 3.
	i = 0
	while i<10:
		Wm,H = NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda)
		W,Hh = NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)

		i += 1
		print i
	print "Step 4. Over..."

	#Z = WH
	Z = W * H
	#filename = r"..\data\Z2.txt"
	#savetxt(filename, around(Z, decimals=2), fmt="%.2f")
	return Z

def B_NMF_Z3(G,E,U,V,x,y,alpha,beta,gamma,mlambda) :
	'B_NMF解决稀疏矩阵问题'
	#将G矩阵分解为X,Y
	X = G[:,:x]
	Y = G[:y,:]
	#将E矩阵分解为P,Q
	P = E[:,:x]
	Q = E[:y,:]
	nx,mx = X.shape
	ny,my = Y.shape
	#D = int(0.1 * min(x,y))
	D = 5
	W = mat(random.rand(nx,D))
	H = mat(random.rand(D,my))

	R = mat(ones((ny,ny)))
	R = diag(diag(R))
	R1 = mat(zeros((ny,nx-ny)))
	R = hstack((R,R1))

	S = mat(ones((mx,mx)))
	S = diag(diag(S))
	S1 = mat(zeros((my-mx,mx)))
	S = vstack((S,S1))

	#Step 1. First NMF of X
	W,Hh = FirstNMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)
	print "Step 1. Over..."

	#Step 2. NMF of Y
	Wm,H = NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda)
	print "Step 2. Over..."

	#Step 3. Subsequent NMF of X
	W,Hh = NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)
	print "Step 3. Over..."

	#Step 4. Repeat Steps 2 and 3.
	i = 0
	while i<10:
		Wm,H = NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda)
		W,Hh = NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)

		i += 1
		print i
	print "Step 4. Over..."

	#Z = WH
	Z = W * H
	#filename = r"..\data\Z3.txt"
	#savetxt(filename, around(Z, decimals=2), fmt="%.2f")
	return Z

def B_NMF_Z4(G,E,U,V,x,y,alpha,beta,gamma,mlambda) :
	'B_NMF解决稀疏矩阵问题'
	#将G矩阵分解为X,Y
	X = G[:,x:]
	Y = G[:y,:]
	#将E矩阵分解为P,Q
	P = E[:,x:]
	Q = E[:y,:]
	nx,mx = X.shape
	ny,my = Y.shape
	#D = int(0.1 * min(x*9,y))
	D = 5
	W = mat(random.rand(nx,D))
	H = mat(random.rand(D,my))

	R = mat(ones((ny,ny)))
	R = diag(diag(R))
	R1 = mat(zeros((ny,nx-ny)))
	R = hstack((R,R1))

	S = mat(ones((mx,mx)))
	S = diag(diag(S))
	S1 = mat(zeros((my-mx,mx)))
	S = vstack((S1,S))

	#Step 1. First NMF of X
	W,Hh = FirstNMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)
	print "Step 1. Over..."

	#Step 2. NMF of Y
	Wm,H = NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda)
	print "Step 2. Over..."

	#Step 3. Subsequent NMF of X
	W,Hh = NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)
	print "Step 3. Over..."

	#Step 4. Repeat Steps 2 and 3.
	i = 0
	while i<10:
		Wm,H = NMFofY(Y,Q,V,D,R,W,H,alpha,beta,gamma,mlambda)
		W,Hh = NMFofX(X,P,U,D,S,W,H,alpha,beta,gamma,mlambda)

		i += 1
		print i
	print "Step 4. Over..."

	#Z = WH
	Z = W * H
	#filename = r"..\data\Z4.txt"
	#savetxt(filename, around(Z, decimals=2), fmt="%.2f")
	return Z

def b_nmf(G,U,V) :
	#E矩阵，user-item矩阵，n*m，0-1矩阵，有意义为1，无意义0
	#E = G
	E = power(G,2)
	#E = exp(G) - 1
	#E = log(G + 1)
	#E = 1.0 / (1 + exp(-G)) - 0.5

	#xl,yh，矩阵规模
	n,m = G.shape
	#print n,m

	#alpha,beta 参数
	alpha = 0.01
	beta = 0.01
	gamma = 10
	mlambda = 1e-8

	x = int(m * 0.9)
	y = int(n * 0.1)
	Z1 = B_NMF_Z1(G,E,U,V,x,y,alpha,beta,gamma,mlambda)
	print "Z1 over..."

	x = int(m * 0.1)
	y = int(n * 0.1)
	Z2 = B_NMF_Z2(G,E,U,V,x,y,alpha,beta,gamma,mlambda)
	print "Z2 over..."
	
	x = int(m * 0.9)
	y = int(n * 0.9)
	Z3 = B_NMF_Z3(G,E,U,V,x,y,alpha,beta,gamma,mlambda)
	print "Z3 over..."
	
	x = int(m * 0.1)
	y = int(n * 0.9)
	Z4 = B_NMF_Z4(G,E,U,V,x,y,alpha,beta,gamma,mlambda)
	print "Z4 over..."

def main() :
	#G矩阵，user-item矩阵，n*m
	filename_G = r"mat.txt"
	G = mat(loadtxt(filename_G))
	print "G get!"

	#U矩阵，user-user矩阵，n*n，[0,1]
	filename_U = r"UserMatrix.txt"
	U = mat(loadtxt(filename_U))
	print "U get!"

	#V矩阵，user-user矩阵，n*n，[0,1]
	filename_V = r"ItemMatrix.txt"
	V = mat(loadtxt(filename_V))
	print "V get!"

	g_mat = b_nmf(G,U,V)

	filename = r"result.txt"
	savetxt(filename, g_mat, fmt="%.2f")


if __name__ == '__main__':
	main()