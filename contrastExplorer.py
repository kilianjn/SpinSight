import holoviews as hv
import panel as pn
import param
import numpy as np


class Tissue:

    def __init__(self, PD, T1, T2):
        self.PD = PD
        self.T1 = T1
        self.T2 = T2
        self.M = 0.
    
    def runPulseSequence(self, seq):
        self.M = seq.run(self)


seqTypes = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery']

class PulseSequence:

    def __init__(self, seqType, TR, TE, FA, TI):
        assert(seqType in seqTypes)
        self.seqType = seqType
        self.TR = TR
        self.TE = TE
        self.FA = np.radians(FA)
        self.TI = TI
    
    def run(self, tissue):
        E1 = np.exp(-self.TR/tissue.T1)
        E2 = np.exp(-self.TE/tissue.T2)
        if self.seqType == 'Spin Echo':
            return tissue.PD * (1 - E1) * E2
        elif self.seqType == 'Spoiled Gradient Echo':
            return tissue.PD * E2 * np.sin(self.FA) * (1 - E1) / (1 - np.cos(self.FA) * E1)
        elif self.seqType == 'Inversion Recovery':
            return tissue.PD * E2 * (1 - 2 * np.exp(-self.TI/tissue.T1) + E1)



class ContrastExplorer(param.Parameterized):
    sequence = param.ObjectSelector(default='Spin Echo', objects=seqTypes)
    TR = param.Number(default=1.0, bounds=(0, 1000.0))
    TE = param.Number(default=1.0, bounds=(0, 100.0))
    FA = param.Number(default=90.0, bounds=(0, 90.0), precedence=-1)
    TI = param.Number(default=1.0, bounds=(0, 1000.0), precedence=-1)
    

    @param.depends('sequence', watch=True)
    def _updateVisibility(self):
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1


    @param.depends('sequence', 'TR', 'TE', 'FA', 'TI')
    def getImage(self):
        nx = ny = 64
        
        tissue1 = Tissue(1.0, 210., 65.)
        tissue2 = Tissue(.95, 1000., 110.)
        
        seq = PulseSequence(self.sequence, self.TR, self.TE, self.FA, self.TI)
        
        for tissue in [tissue1, tissue2]:
            tissue.runPulseSequence(seq)
            
        M = np.zeros((nx, ny), dtype=complex)
        M[20:45, 10:50] = tissue1.M
        M[40:60, 30:55] = tissue2.M
        img = hv.Image(abs(M), bounds=[0,0,nx,ny], kdims=['x', 'y'], vdims=['magnitude']).options(aspect='equal', cmap='gray')
        return img

explorer = ContrastExplorer(name='MR Contrast Explorer')
dmapMRimage = hv.DynamicMap(explorer.getImage).opts(framewise=True, frame_height=300)
dashboard = pn.Column(pn.Row(pn.panel(explorer.param), dmapMRimage))
dashboard.show()
