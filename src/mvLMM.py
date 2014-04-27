
# mvLMM is a python-based linear mixed-model solver for multiple phenotypes.

# Copyright (C) 2014  Nicholas A. Furlotte (nick.furlotte@gmail.com)

#The program is free for academic use. Please contact Nick Furlotte
#<nick.furlotte@gmail.com> if you are interested in using the software for
#commercial purposes.

#The software must not be modified and distributed without prior
#permission of the author.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

TRACE = True

# This is needed to limit the number of threads MKL is using
# You should really set this outside of this but I'm being extra paranoid as to not get banned from the cluster again
import os
if not os.environ.has_key("MKL_NUM_THREADS"): os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import numpy as np
from scipy import linalg
import scipy.optimize as optimize
from scipy import stats
from pylmm.lmm import LMM
import pdb


class mvLMM:
   """
      Implements the matrix-variate linear mixed-model (mvLMM).
      To start I will just assume that we only consider two phenotypes at a time and then will expand from that.

      The optimization works by first fitting a LMM for each phenotype and then by holding those parameters constant and fitting the additional parameters.

   """

   def __init__(self,Y,K,Kva=[],Kve=[],norm=True,X0=None,verbose=True):

      # We are assuming that X0 is the same for each phenotype

      self.verbose = verbose

      if len(Kva) == 0 or len(Kve) == 0: Kva,Kve = self.getEigen(K)
      self.K = K
      self.Kva = Kva
      self.Kve = Kve
      if sum(self.Kva <= 0): 
	 sys.stderr.write("Cleaning %d eigen values\n" % (sum(self.Kva < 0)))
	 self.Kva[self.Kva <= 0] = 1e-6

      
      self.Y = self.cleanPhenos(Y)
      if norm: self.Y = self.normPhenos(self.Y)
      self.N = Y.shape[0]
      self.M = Y.shape[1]
      if X0 == None: X0 = self._getDefaultX0()
      self.X0 = X0

      self.LMMs = []
      for i in range(self.M): 
	 self.LMMs.append(LMM(Y[:,i],self.K,self.Kva,self.Kve,X0=self.X0))
	 # Fitting under the NULL where the SNP has no effect
	 self.LMMs[i].fit()
      
      self._cacheLLStuff()

      self.mxCor = None
      self.R = None

      self.gcors = None
      self.ecors = None
      self.ngrids = 100
      self._setDefaultGcorsandEcors(self.ngrids)

   def _setDefaultGcorsandEcors(self,ngrids=100):
      self.ngrids = 100
      gcors = np.array(range(ngrids))/float(ngrids)
      gcors = gcors[1:-1]
      gcors2 = (-1.0 * gcors.copy()).tolist()
      gcors2.reverse()
      gcors = gcors2 + gcors.tolist()

      self.gcors = np.array(gcors)
      self.ecors = self.gcors.copy()

   def _getDefaultX0(self):
      x = np.ones((self.N,1))
      return x

   def _eigHForKva(self):
      L = 1.0 / self.Kva
      L.sort()
      M = np.ones((self.N,self.N))*0.0
      for i in range(self.N): M[self.N-1-i,i] = 1.0
      return L,M
      
   def getEigen(self,K):
      sys.stderr.write("Obtaining eigendecomposition for %dx%d matrix\n" % (K.shape[0],K.shape[1]) )
      begin = time.time()
      Kva,Kve = linalg.eigh(K)
      end = time.time()
      sys.stderr.write("Total time: %0.3f\n" % (end - begin))
      return Kva,Kve

   def normPhenos(self,Y):
      M = Y.shape[1]
      for i in range(M): Y[:,i] = (Y[:,i] - Y[:,i].mean())/np.sqrt(Y[:,i].var())
      return Y

   def cleanPhenos(self,Y):
      M = Y.shape[1] 
      for i in range(M):
	 y = Y[:,i]
	 x = True - np.isnan(y)
	 if sum(x) == len(y): continue
	 m = y[x].mean()
	 y[np.isnan(y)] = m
	 Y[:,i] = y
      return Y

   def _cacheLLStuff(self):
      self.leftTransform = self.Kve.T
      self.X0_T = np.dot(self.leftTransform,self.X0)
      
      self.Ystar = np.dot(self.leftTransform, self.Y)

   def _getBetaT_original(self,XStar,Ap,P,Yt):
      L = np.kron(Ap,XStar)
      A = L.T * 1.0/(P+1.0)
      B = np.dot(A,L)
      Bi = linalg.inv(B)
      beta = np.dot(np.dot(Bi,A),Yt)

      mu = np.dot(L,beta)
      return beta,mu,Bi

   def _getBetaT(self,XStar,Ap,D,Yt):
      P = []
      for i in range(self.M): P += (self.Kva + D[i]).tolist()
      P = np.array(P)
      L = np.kron(Ap,XStar)
      A = L.T * 1.0/(P+1.0)
      B = np.dot(A,L)
      Bi = linalg.inv(B)
      beta = np.dot(np.dot(Bi,A),Yt)

      _REML_part = np.log(linalg.det(np.dot(L.T,L))) + np.log(linalg.det(B))

      mu = np.dot(L,beta)
      return beta,mu,Bi,_REML_part

   def getParameterMatrices(self,gcor,ecor): 

      Psi = np.ones((self.M,self.M))
      Phi = np.ones((self.M,self.M))

      for i in range(self.M):
	 Psi[i,i] = self.LMMs[i].optH*self.LMMs[i].optSigma
	 Phi[i,i] = (1.0 - self.LMMs[i].optH)*self.LMMs[i].optSigma
      for i in range(self.M - 1):
	 for j in range(i+1,self.M):
	    Psi[i,j] = Psi[j,i] = np.sqrt(Psi[i,i])*np.sqrt(Psi[j,j]) * gcor[i,j]
	    Phi[i,j] = Phi[j,i] = np.sqrt(Phi[i,i])*np.sqrt(Phi[j,j]) * ecor[i,j]

      return Psi,Phi

   def LL(self,gcor,ecor, X=None,REML=False,computeHard=False):
      return self.LL_ML(gcor,ecor,X,computeHard=computeHard,REML=REML)

   def LL_REML(self,gcor,ecor,computeHard=False): 
      LL = self.LL_ML(gcor,ecor,X=X,computeHard=computeHard)

      pdb.set_trace()
      l = LL[0]

   def LL_ML(self,gcor,ecor,X=None,computeHard=False,trace=False,REML=False):

      if trace: pdb.set_trace()

      Psi, Phi = self.getParameterMatrices(gcor,ecor)

      # Check that they are positive semi-def
      Psi_Kva,Psi_Kve = linalg.eigh(Psi)
      Phi_Kva,Phi_Kve = linalg.eigh(Phi)

      Psi_Kva[Psi_Kva == 0] = 1e-6
      Phi_Kva[Phi_Kva == 0] = 1e-6

      if sum(Psi_Kva < 0) or sum(Phi_Kva < 0):
	 #sys.stderr.write("Negative Eigen Values: \n %s \n %s\n" % (str(gcor),str(ecor)))
	 #sys.stderr.write("Negative Eigen Values at %0.3f/%0.3f\n" % (gcor,ecor))
	 return [np.nan]


      # Get the D matrix by diagonalizing Psi and Phi 
      R = Psi_Kve*np.sqrt(1.0/Psi_Kva) # Now R %*% R.T is = Psi^{-1}
      RR = np.dot(np.dot(R.T,Phi),R)
      RR_Kva,RR_Kve = linalg.eigh(RR)
      D = RR_Kva
      rightTransform = np.dot(RR_Kve.T,R.T)

      # Get Transformed Y
      Yt = np.dot(self.Ystar,rightTransform.T).T.reshape(-1,1)

      if X == None: X = self.X0_T
      else: X = np.hstack([self.X0_T, np.dot(self.leftTransform,X)])

      beta_T,mu,beta_T_stderr,_REML_part = self._getBetaT(X,rightTransform,D,Yt)
      Yt = Yt - mu

      # Now get the LL
      C = 0.0
      for i in range(self.M):
	 #for j in range(self.N): C += np.log(self.Kva[j] + D[i])
	 C += np.log(self.Kva + D[i]).sum()
      Yc = Yt.T.copy()
      for i in range(self.M): Yc[0,i*self.N:i*self.N+self.N] = Yc[0,i*self.N:i*self.N+self.N] * (1.0 / (self.Kva + D[i]))
      Q = np.dot(Yc,Yt)
      sigma = Q / float(len(Yt) - len(beta_T))

      # Assumes that residual error is 1.0
      #LL = float(self.N)*float(self.M)*np.log(2.0*np.pi) + C + Q

      # Incorporates resisdual error - estmated covariance matrix is not perfect
      LL = float(self.N)*float(self.M)*np.log(2.0*np.pi) + C + float(self.N)*float(self.M)*np.log(sigma)


      # Incorporates resisdual error - estmated covariance matrix is not perfect
      #LL = float(self.N)*float(self.M)*np.log(2.0*np.pi) + C + float(self.N)*float(self.M) + float(self.N)*float(self.M)*np.log(1.0/(float(self.N)*float(self.M)) * Q)

      LL = -0.5 * LL

      # normalization term -- ignoring the constants from K
      # the det of the righttransform is simplified because it is a function of eigenvectors
      logM = float(self.N) * np.log(np.sqrt(1.0/Psi_Kva)).sum()
      #logM = float(self.N) * np.log(np.linalg.det(rightTransform))

      LL = LL + logM

      if REML: 
	 #sys.stderr.write("Calculating REML\n")
	 LL_REML_part =  len(beta_T) * np.log(2.0*np.pi*sigma) + _REML_part
	 LL = LL + 0.5 * LL_REML_part

      # The hard way is the absolute naive way which takes a very long time
      # This is just an option for sanity/error checking
      if computeHard:
	 sys.stderr.write("Computing hard version\n")
	 Sigma = np.kron(Psi,self.K) + np.kron(Phi,np.diag(np.ones(self.N)))
	 Sigma_inv = linalg.inv(Sigma)
	 y = self.Y.T.reshape(-1,1)
	 LL_hard_C = np.log(linalg.det(Sigma))
	 LL_hard_Q = np.dot(np.dot(y.T,Sigma_inv),y)
	 LL_hard = float(self.N)*2.0*np.log(2.0*np.pi) + LL_hard_C + LL_hard_Q
	 LL_hard = -0.5 * LL_hard
	 return LL_hard,beta_T,sigma,beta_T_stderr

      # beta_T_stderr is actually the variance of the estimator
      # It is just hard to change it around all the code right now...
      return LL,beta_T,sigma,beta_T_stderr

   def LL_ML_original(self,gcor,ecor,X=None,computeHard=False,REML=False):

      # again written assuming only two phenotypes
      assert(self.M == 2)

      #Psi, Phi = self.getParameterMatrices(gcor,ecor)
      sg1 = np.sqrt(self.LMMs[0].optH*self.LMMs[0].optSigma)
      sg2 = np.sqrt(self.LMMs[1].optH*self.LMMs[1].optSigma)
      Psi = np.array([[sg1**2,sg1*sg2*gcor],[sg1*sg2*gcor,sg2**2]])

      se1 = np.sqrt((1.0 - self.LMMs[0].optH)*self.LMMs[0].optSigma)
      se2 = np.sqrt((1.0 - self.LMMs[1].optH)*self.LMMs[1].optSigma)
      Phi = np.array([[se1**2,se1*se2*ecor],[se1*se2*ecor,se2**2]])

      Psi_Kva,Psi_Kve = linalg.eigh(Psi)
      Phi_Kva,Phi_Kve = linalg.eigh(Phi)

      Psi_Kva[Psi_Kva == 0] = 1e-6
      Phi_Kva[Phi_Kva == 0] = 1e-6

      if sum(Psi_Kva < 0) or sum(Phi_Kva < 0):
	 sys.stderr.write("Negative Eigen Values at %0.3f/%0.3f\n" % (gcor,ecor))
	 return [np.nan]

      # Get the P matrix
      R = Psi_Kve*np.sqrt(1.0/Psi_Kva)
      RR = np.dot(np.dot(R.T,Phi),R)
      RR_Kva,RR_Kve = linalg.eigh(RR)
      P = []
      for i in range(len(RR_Kva)):
	 for j in range(len(self.Kva_Kva)): P.append(RR_Kva[i] * self.Kva_Kva[j])
      P = np.array(P)

      # get transformed Y : Yt = Mvec(Y) =   Q^\prime L vec(Y)
      # We have cached self.M_1
      D = np.dot(R,RR_Kve)
      Yt = np.dot(self.M_1,D).T.reshape(-1,1)
      #D = np.dot(self.M_1,R)
      #Yt = np.dot(D,RR_Kve).T.reshape(-1,1)


      if X == None: X = self.X0_T
      else: X = np.hstack([self.X0_T, np.dot(self.leftTransform,X)])

      beta_T,mu,beta_T_stderr = self._getBetaT_original(X,D.T,P,Yt)
      Yt = Yt - mu

      # Now get the LL
      C = np.log(P+1.0).sum()
      Q = np.dot(Yt.T*(1.0/(P+1.0)),Yt)
      sigma = Q / float(len(Yt) - len(beta_T))

      # Assumes that residual error is 1.0
      LL = float(self.N)*float(self.M)*np.log(2.0*np.pi) + C + Q

      # Incorporates resisdual error - estmated covariance matrix is not perfect
      #LL = float(self.N)*float(self.M)*np.log(2.0*np.pi) + C + float(self.N)*float(self.M) + float(self.N)*float(self.M)*np.log(1.0/(float(self.N)*float(self.M)) * Q)

      LL = -0.5 * LL

      # normalization term -- ignoring the constants from K
      logM = float(self.N) * np.log(np.sqrt(1.0/Psi_Kva)).sum()

      LL = LL + logM

      # The hard way is the absolute naive way which takes a very long time
      # This is just an option for sanity/error checking
      if computeHard:
	 sys.stderr.write("Computing hard version\n")
	 Sigma = np.kron(Psi,self.K) + np.kron(Phi,np.diag(np.ones(self.N)))
	 Sigma_inv = linalg.inv(Sigma)
	 y = self.Y.T.reshape(-1,1)
	 LL_hard_C = np.log(linalg.det(Sigma))
	 LL_hard_Q = np.dot(np.dot(y.T,Sigma_inv),y)
	 LL_hard = float(self.N)*2.0*np.log(2.0*np.pi) + LL_hard_C + LL_hard_Q
	 LL_hard = -0.5 * LL_hard
	 return LL_hard

      # beta_T_stderr is actually the variance of the estimator
      # It is just hard to change it around all the code right now...
      return LL,beta_T,sigma,beta_T_stderr

   def fit_multiple(self,ngrids=10,computeHard=False,type='1111',REML=False):

      cors = (np.array(range(ngrids))/float(ngrids)).tolist()
      ncors = [-x for x in cors[1:]]
      ncors.reverse()
      tcors = ncors + cors    
   
      self.gcors = np.array(tcors)
      self.ecors = self.gcors.copy()

      assert(self.M == 3)

      mx = None
      mxCor = None

      pdb.set_trace()

      M = mvLMM(self.Y[:,[0,1]],self.K,self.Kva,self.Kve)
      M.getMaxWithCourseSearch()
      gcor01 = M.mxCor[0]
      ecor01 = M.mxCor[1]

      M = mvLMM(self.Y[:,[1,2]],self.K,self.Kva,self.Kve)
      M.getMaxWithCourseSearch()
      gcor12 = M.mxCor[0]
      ecor12 = M.mxCor[1]

      for gcor02 in self.gcors:
	 for ecor02 in self.ecors:
	   gpoint = np.array([[1.0, gcor01, gcor02],[gcor01,1.0,gcor12],[gcor02,gcor12,1.0]])
	   epoint = np.array([[1.0, ecor01, ecor02],[ecor01,1.0,ecor12],[ecor02,ecor12,1.0]])
	   X = self.LL(gpoint,epoint,computeHard=computeHard)[0]
	   if np.isnan(X): continue
	   if mx == None or X > mx: 
	     mx = X
	     mxCor = ([gcor01,gcor02,gcor12],[ecor01,ecor02,ecor12])

      return mx,mxCor

   def fit(self,ngrids=100,computeHard=False,type='1111',REML=False):
      """
	 type is set to tell the function the sign of the correlation parameters to try.
	 The first bit sets gcor as positive and ecor as positive.  The second sets them both to negative.
	 The third sets gcor to positive and ecor to negative and the fourth the opposite.
	 In general, without previous information you probably want to evaluate the full space, but sometimes 
	 you might just want to see a specific case to save time.

      """

      gcors = np.array(range(ngrids))/float(ngrids)
      gcors = gcors[1:-1]
      ecors = np.array([x for x in gcors])

      self.gcors = gcors
      self.ecors = ecors

      sys.stderr.write("Starting model fit...\n")
      begin = time.time()

      if type[0] == '1': 
	 self.R_pos,self.mx_pos,self.mxCor_pos,self.mxIn_pos = self._fit(gcors,ecors,computeHard=computeHard,REML=REML)
      else: self.R_pos,self.mx_pos,self.mxCor_pos,self.mxIn_pos = tuple([None for i in range(4)])

      if type[1] == '1': self.R_neg,self.mx_neg,self.mxCor_neg,self.mxIn_neg = self._fit(-gcors,-ecors,computeHard=computeHard,REML=REML)
      else: self.R_neg,self.mx_neg,self.mxCor_neg,self.mxIn_neg = tuple([None for i in range(4)])

      if type[2] == '1': self.R_posneg,self.mx_posneg,self.mxCor_posneg,self.mxIn_posneg = self._fit(gcors,-ecors,computeHard=computeHard,REML=REML)
      else: self.R_posneg,self.mx_posneg,self.mxCor_posneg,self.mxIn_posneg = tuple([None for i in range(4)])

      if type[3] == '1': self.R_negpos,self.mx_negpos,self.mxCor_negpos,self.mxIn_negpos = self._fit(-gcors,ecors,computeHard=computeHard,REML=REML)
      else: self.R_negpos,self.mx_negpos,self.mxCor_negpos,self.mxIn_negpos = tuple([None for i in range(4)])

      sys.stderr.write("Completed in %0.3f seconds\n" % (time.time() - begin))

   def _fit(self,gcors,ecors,computeHard=False,REML=False):
      ngrids = len(gcors)
      assert(ngrids == len(ecors))
      R = np.ones((ngrids,ngrids))*np.nan
      mx = None
      mxIn = None
      mxCor = None
      #D = []

      gcor_matrix = np.ones((self.M,self.M)) * np.nan
      ecor_matrix = np.ones((self.M,self.M)) * np.nan

      for i in range(len(gcors)):
	 for j in range(len(ecors)): 
	    gcor_matrix[0,1] = gcor_matrix[1,0] = gcors[i]
	    ecor_matrix[0,1] = ecor_matrix[1,0] = ecors[j]
	    R[i,j] = self.LL(gcor_matrix,ecor_matrix,computeHard=computeHard,REML=REML)[0]

	    #R[i,j] = self.LL(gcors[i],ecors[j],computeHard=computeHard,REML=REML)[0]
	    if mx == None or R[i,j] > mx: 
	       #if not mx == None: 
		  #d = mx - R[i,j]
		  #D.append((gcors[i],ecors[j],d))
	       mx = R[i,j]
	       mxIn = (i,j)
	       mxCor = (gcors[i],ecors[j])
	       

      return R,mx,mxCor,mxIn
      #return R,mx,mxCor,mxIn,D

   def normMatrix(self,R):
      R_norm = R - R.max()
      R_norm = np.exp(R_norm)
      R_norm = R_norm / R_norm.sum()
      return R_norm

   def plotFullGrid(self,norm=True,xticks=[],yticks=[]):

      import matplotlib.pyplot as pl
      R = np.hstack([self.R_neg, self.R_negpos])
      R1 = np.hstack([self.R_posneg, self.R_pos])
      R = np.vstack([R,R1])
      if norm: R_norm = self.normMatrix(R)
      else: R_norm = R

      p = pl.pcolor(R_norm)

      ngrids = R.shape[0] + 4
      f = p.get_axes()
      if not len(xticks): xticks = f.xaxis.get_ticklocs() / float(ngrids)
      if not len(yticks): yticks = f.yaxis.get_ticklocs() / float(ngrids)

      xticks[0:int(len(xticks)/2.0)]  = -xticks[0:int(len(xticks)/2.0)]
      yticks[0:int(len(yticks)/2.0)]  = -yticks[0:int(len(yticks)/2.0)]
      pl.xticks(f.xaxis.get_ticklocs(), xticks)
      pl.yticks(f.yaxis.get_ticklocs(), yticks)

      pl.xlabel("Environmental Correlation")
      pl.ylabel("Genetic Correlation")

   def plotFit(self,R = [],norm=True,xticks=[],yticks=[],a=1,b=1):
      import matplotlib.pyplot as pl
      if not len(R): return self.plotPosterior()
     
      if norm: R_norm = self.normMatrix(R) 
      else: R_norm = R
      p = pl.pcolor(R_norm)
      # Assume that R was calculated with ngrids while removing the first 
      # and last entries as is done in fit.  recalculate ngrids
      # THIS IS A HACK
      # I need a more consistent way of formatting the ticks.
      # This will not work when I resize for example
      ngrids = R.shape[0] + 2
      f = p.get_axes()
      if not len(xticks): xticks = f.xaxis.get_ticklocs() / float(ngrids)
      if not len(yticks): yticks = f.yaxis.get_ticklocs() / float(ngrids)
      pl.xticks(f.xaxis.get_ticklocs(), a*xticks)
      pl.yticks(f.yaxis.get_ticklocs(), b*yticks)
      pl.xlabel("Environmental Correlation")
      pl.ylabel("Genetic Correlation")

   def surfacePlot(self,X,Y,Z,fig=None):
	 # For the surface plot
	 import matplotlib.pyplot as pl
	 from mpl_toolkits.mplot3d import Axes3D
	 from matplotlib import cm
	 from matplotlib.ticker import LinearLocator, FormatStrFormatter

	 if not fig: fig = pl.figure()
	 ax = fig.gca(projection='3d')
	 X, Y = np.meshgrid(X, Y)
	 surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
	 ax.set_zlim(0.00,Z.max())

	 ax.zaxis.set_major_locator(LinearLocator(10))
	 ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	 fig.colorbar(surf, shrink=0.5, aspect=5)

	 pl.xlabel("Environmental Correlation")
	 pl.ylabel("Genetic Correlation")

   def getMaxWithCourseSearch(self,ngrids=50,courseGrids=10,computeHard=False,REML=False):
	 """
	 Finds the ML correlations by first conducting a course grid search over the full space.  Then by finding the quadrant that has the maximum and doing a dense search within there to clarify.
	 It sets self.mxCor and self.R to the results so that this can be used programatically in association analysis.
	 """

	 self.fit(ngrids=courseGrids,computeHard=computeHard,REML=REML)
	 x = self.R_pos.max()
	 type = '1000'
	 y = self.R_neg.max()
	 if y > x:
	    x = y
	    type = '0100'

	 z = self.R_posneg.max()
	 if z > x:
	    x = z
	    type = '0010'

	 q = self.R_negpos.max()
	 if q > x:
	    x = q
	    type = '0001'
	 sys.stderr.write("Identified %s type\n" % type)

	 self.fit(ngrids=ngrids,type=type,computeHard=computeHard,REML=REML)
	 if type == '1000':  X,Y,R,mxCor = self.gcors,self.ecors,self.R_pos,self.mxCor_pos
	 elif type == '0100': X,Y,R,mxCor = -self.gcors,-self.ecors,self.R_neg,self.mxCor_neg
	 elif type == '0010': X,Y,R,mxCor = self.gcors,-self.ecors,self.R_posneg,self.mxCor_posneg
	 elif type == '0001': X,Y,R,mxCor = -self.gcors,self.ecors,self.R_negpos,self.mxCor_negpos

	 self.mxCor = mxCor
	 self.R = R

   def getMax(self,ngrids=100,REML=False,tripletSearch=True):
      gcors = np.array(range(ngrids))/float(ngrids)
      gcors = gcors[1:-1]
      gcors2 = (-1.0 * gcors.copy()).tolist()
      gcors2.reverse()
      gcors = gcors2 + gcors.tolist()
      self.gcors = np.array(gcors)
      self.ecors = self.gcors.copy()

      if not len(self.gcors): return np.nan

      if self.verbose: sys.stderr.write("Starting model fit...\n")
      begin = time.time()

      R = []
      mx = None
      mxIn = None
      mxCor = None

      R,mx,mxCor,mxIn = self._fit(self.gcors,self.ecors,REML=REML)

      # The idea here is to look for peaks in two dimensions only if 
      # one of the solutions is on the boundary
      if tripletSearch and (np.abs(mxCor[0]) >= 0.90 or np.abs(mxCor[1]) >= 0.90):
	 mxTrip = None
	 if self.verbose: sys.stderr.write("Trying triplets with mxCor=%s...\n" % (str(mxCor)))
	 for i in range(1,len(self.gcors)-2): 
	    for j in range(1,len(self.ecors)-2): 
	       r = R[[i-1,i,i+1],:][:,[j-1,j,j+1]]
	       if r.max() == R[i,j]: 
		  if self.verbose: sys.stderr.write("Found Triplet at %0.2f / %0.2f with LL = %0.5f\n" % (self.gcors[i],self.ecors[j],R[i,j]))
		  if mxTrip == None or mxTrip[2] < R[i,j]:
		     mxTrip = (self.gcors[i],self.ecors[j],R[i,j])
	 
	 if not mxTrip == None:
	    mxCor = (mxTrip[0],mxTrip[1])
	    mx = mxTrip[2]

      end = time.time()

      if self.verbose: sys.stderr.write("Total fit time: %0.3f\n" % (end - begin))

      self.mxCor = mxCor
      self.R = R
      self.mx = mx
      self.mxIn = mxIn

      return R,mx,mxCor,mxIn

   def summaryAnalysis(self,title='',ngrids=50,courseGrids=10,computeHard=False):
	 import matplotlib.pyplot as pl
	 fig = pl.figure(figsize=(13,6))

	 pl.subplot(121)
	 self.fit(ngrids=courseGrids,computeHard=computeHard)
	 self.plotFullGrid()
	 pl.title(title + ' (Course Grid)')

	 x = self.R_pos.max()
	 type = '1000'
	 y = self.R_neg.max()
	 if y > x:
	    x = y
	    type = '0100'

	 z = self.R_posneg.max()
	 if z > x:
	    x = z
	    type = '0010'

	 q = self.R_negpos.max()
	 if q > x:
	    x = q
	    type = '0001'

	 sys.stderr.write("Identified %s type\n" % type)
	 self.fit(ngrids=ngrids,type=type,computeHard=computeHard)
	 if type == '1000':  X,Y,R,mxCor = self.gcors,self.ecors,self.R_pos,self.mxCor_pos
	 elif type == '0100': X,Y,R,mxCor = -self.gcors,-self.ecors,self.R_neg,self.mxCor_neg
	 elif type == '0010': X,Y,R,mxCor = self.gcors,-self.ecors,self.R_posneg,self.mxCor_posneg
	 elif type == '0001': X,Y,R,mxCor = -self.gcors,self.ecors,self.R_negpos,self.mxCor_negpos

	 R = self.normMatrix(R)

	 pl.subplot(122)
	 self.plotFit(R,norm=False,a = (Y[-1] > 0 and 1 or -1), b=(X[-1] > 0 and 1 or -1))
	 pl.title(title + " (2D Detailed Fit) \n Max = %0.3f,%0.3f" % mxCor)

	 self.surfacePlot(Y,X,R)
	 pl.title(title) #  + " (3D Detailed Fit)")

   def association(self,X, gcor=None, ecor=None,REML=False):
      if (not gcor or not ecor) and (not self.mxCor): self.getMaxWithCourseSearch()
      if not gcor: gcor = self.mxCor[0]
      if not ecor: ecor = self.mxCor[1]

      L,beta,sigma,betaSTDERR = self.LL(gcor,ecor,X,REML=REML)
      q  = len(beta)
      fs,ps = self.fstat(beta,betaSTDERR,sigma,q/self.M)
      return fs.sum(),ps.sum(),beta[q-1].sum(),betaSTDERR[q-1,q-1]

   def fstat(self,beta,stderr,sigma,q): 
	 # Assumes that the tested effect is in the q-1 position for 
	 # column in beta.
	 # This is what happens if a SNP vector is passed to .association.
	 p = self.M
	 R = [[0.0]*(p*q) for i in range(p)]
	 for i in range(p): R[i][i*q+(q-1)] = 1.0
	 R = np.array(R)

	 Rb = np.dot(R,beta)
	 V = np.dot(np.dot(R,stderr),R.T)
	 Vi = linalg.inv(V)
	 fs = 1.0/(sigma*p) * np.dot(np.dot(Rb.T,Vi),Rb)
	 ps = 1.0 - stats.f.cdf(fs,p,(self.N*self.M) - (self.M*q))

	 return fs,ps
   
   def stat(self): pass

       #P = R %*% betaHat$beta
       #Pt = t(P)
       #fs = Pt %*% solve((R %*% betaHat$stderrSQ %*% Rt)*sigmaHat) %*% P  # again need to consider residual error of Y after transformation
       #fs = fs / p
       #ps = pf(fs, p, N-p*(q+1), lower.tail=F)
	 
   def getMeanAndVariance(self,R):
      Rn = self.normMatrix(R)
      Ex = (self.gcors * Rn.sum(1)).sum() 
      X_Ex = (Rn.sum(1)*(self.gcors - Ex)**2).sum()
      return Ex,X_Ex
     
       
      

      

	 
      

      
   
      
      
      



