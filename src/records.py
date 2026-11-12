# what the forward pass stores per layer, and what the backward pass returns per layer
import numpy as np


class LayerOutput:
    def __init__(self, z=np.array([]), act=np.array([]), nm=""):
        self.z = z
        self.activation = act
        self.nm = nm

    def __repr__(self):
        return f"LayerOutput: z -> {self.z.shape}; activation -> {self.activation.shape}; type -> {self.nm}"


class LayerGrad:
    def __init__(self, dw=np.array([]) , db=np.array([]), dz=np.array([]), d_act=np.array([]), nm=""):
        self.dw = dw
        self.db = db
        self.dz = dz
        self.d_act = d_act
        self.nm = nm

    def __repr__(self):
        return f"LayerGrad: dw -> {self.dw.shape}; db -> {self.db.shape}; dz -> {self.dz.shape}; d_act -> {self.d_act.shape}; type -> {self.nm}"


class LayerOutputBN(LayerOutput):
    def __init__(self, z=np.array([]), z_bn=np.array([]), x_hat=np.array([]), var=np.array([]), act=np.array([]), nm=""):
        super().__init__(z, act, nm)
        self.z_bn = z_bn          # BatchNorm output, the input to ReLU
        self.x_hat = x_hat        # standardised z, needed by eq. (45)
        self.var = var            # batch variance, needed by eq. (45)

    def __repr__(self):
        return f"LayerOutputBN: z -> {self.z.shape}; z_bn -> {self.z_bn.shape}; activation -> {self.activation.shape}; type -> {self.nm}"


class LayerGradBN(LayerGrad):
    def __init__(self, dw=np.array([]), db=np.array([]), dz=np.array([]), d_act=np.array([]), dgamma=np.array([]), dbeta=np.array([]), nm=""):
        super().__init__(dw, db, dz, d_act, nm)
        self.dgamma = dgamma
        self.dbeta = dbeta

    def __repr__(self):
        return f"LayerGradBN: dw -> {self.dw.shape}; dgamma -> {self.dgamma.shape}; dbeta -> {self.dbeta.shape}; type -> {self.nm}"
