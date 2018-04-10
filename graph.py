import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.utils.extmath import randomized_svd
from scipy import sparse

				
class Data(object): # parent class than handles PCA / import of data
	def __init__(self, data, ndim, random_state):
		self.data = data
		if ndim > 0 and ndim < data.shape[1]:
			pca = PCA(ndim, svd_solver='randomized',random_state=random_state);
			self.data_nu = pca.fit_transform(data)
			self.U = pca.components
			self.S = pca.singular_values
		else:
			self.data_nu = data

class ParentGraph(object):  # all graphs should possess these matrices
	def __init__(self,*args):
		self.K = None
		self.build_K(*args)
		self._A = None
		self._d = None
		self._diffop = None
		self._L = None
		self.lap_type = "combinatorial"

	@property 
	def A(self):
		if not self._A:
			self._A = (self.K + self.K.T)/2
		if self._A[1,1] > 0:
			self._A = np.fill_diagonal(self._A,0)
		return self._A

	@property
	def D(self):
		if not self._d:
			self._d = self.A.sum(axis=1)[:, None]
		return self._d

	@property
	def diffop(self):
		if not self._diffop:
			self._diffop = self.K / self.D
		return self._diffop

	@property
	def L(self, lap_type = "combinatorial"):
		if not self._L or lap_type is not self.lap_type:
			if lap_type is "combinatorial":
				self._L = self.D - self.A
			elif lap_type is "normalized":
				self._L = np.eye(self.A.shape[0]) - np.diag(self.D)^-5 * self.A * np.diag(self.D)^-.5
			elif lap_type is "randomwalk":
				self._L = np.eye(self.A.shape[0]) - self.diffop
		return self._L

class kNNGraph(ParentGraph, Data): #build a kNN graph
	def __init__(self, data, ndim = 0, knn = 5, decay = 0,  thresh = 1e-5, random_state = None, distance = 'Euclidean'):
		Data.__init__(self, data, ndim, random_state)
		ParentGraph.__init__(self, k, decay, thresh, distance)

	def build_K(self, knn, decay, thresh, distance):
		if decay == 0:
			self.K = kneighbors_graph(self.data_nu,knn, mode='connectivity',include_self=True ,metric = distance)
		else:
			kalpha = knn
			tmp= kneighbors_graph(self.data_nu, kalpha, mode='distance', include_self = False ,metric = distance)
			bandwidth = sparse.diags(1/np.max(tmp,1).A.ravel())
			ktmp = np.exp(-1*(tmp*bandwidth)**decay)
			while (np.min(ktmp[np.nonzero(ktmp)])>thresh):
				knn += 5
				tmp= kneighbors_graph(self.data_nu, knn, mode='distance', include_self = False, metric = distance)
				ktmp = np.exp(-1*(tmp*bandwidth)**decay)
			self.K = ktmp

class TraditionalGraph(ParentGraph, Data):
	def __init__(self, data, ndim, decay = 10, knn = 5, precomputed = None, thresh = None, random_state = None, distance = 'Euclidean')
		if precomputed is None:
			ndim = 0
		Data.__init__(self,data, ndim, random_state)
		ParentGraph.__init__(self, decay, knn, precomputed, distance)

	def build_K(self, decay, k, precomputed, distance)
		if precomputed is "distance":
			pdx = self.data_nu;
		if precomputed is None:
			pdx = squareform(pdist(data, metric=distance))
		if precomputed is not "affinity":
            knn_dist = np.partition(pdx, k, axis=1)[:, :k]
        	epsilon = np.max(knn_dist,axis=1);
        	pdx = (pdx/epsilon).T
        	self.K = np.exp(-1 * pdx**decay)
        else:
        	self.K = self.data_nu

    	self.K = self.K + self.K.T




def Graph(graphtype, data, *args, **kwargs):
	if graphtype == "knn":
		base = kNNGraph
	elif graphtype == "exact":
		base = TraditionalGraph

	class Graph(base):
		def __init__(self, data, *args, **kwargs):
			base.__init__(self, data, *args, **kwargs)
	

	return Graph(data,*args,**kwargs)



