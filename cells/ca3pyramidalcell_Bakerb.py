# CA3 Pyramidal Cell Class
from neuron import h
import pickle
#h.xopen("ca3b.hoc")

# The following layers are dictionaries whose keys are NEURON sections.
# Each entry of the dictionaries contains a list. The list contains
# the normalized distances along the section that lie within a layer.
def makeSecDict():
	SecList = {}
	SecList['oriensDistal'] = {}
	SecList['oriensProximal'] = {}
	SecList['soma'] = {}
	SecList['lucidum'] = {}
	SecList['radiatum'] = {}
	SecList['lacunosumMEC'] = {}
	SecList['lacunosumLEC'] = {}
	return SecList

def makeLayerDict(cell):
	LayerDict = {}
	LayerDict['Apical'] = {}
	LayerDict['Apical']['soma'] = cell.soma
	LayerDict['Apical']['lucidum'] = cell.lucidum
	LayerDict['Apical']['radiatum'] = cell.radiatum
	LayerDict['Apical']['lacunosumMEC'] = cell.lacunosumMEC
	LayerDict['Apical']['lacunosumLEC'] = cell.lacunosumLEC
	
	LayerDict['Basal'] = {}
	LayerDict['Basal']['oriensProximal'] = cell.oriensProximal
	LayerDict['Basal']['oriensDistal'] = cell.oriensDistal
	return LayerDict


# Since all the morphology is defined by HOC code, we need a pointer to the HOC object.
def loadMorph(morphFileName):
	param = {}
	param['c'] = h.ca3Cell()
	return param

# Initilize the list of synapses for the cell
def makeSynGroups(cell):
	SynGroups = {}
	SynGroups['AMPA'] = {}
	SynGroups['AMPA']['oriensDistal'] = []
	SynGroups['AMPA']['oriensProximal'] = []
	SynGroups['AMPA']['soma'] = []
	SynGroups['AMPA']['lucidum'] = []
	SynGroups['AMPA']['radiatum'] = []
	SynGroups['AMPA']['lacunosumMEC'] = []
	SynGroups['AMPA']['lacunosumLEC'] = []
	
	SynGroups['NMDA'] = {}
	SynGroups['NMDA']['oriensDistal'] = []
	SynGroups['NMDA']['oriensProximal'] = []
	SynGroups['NMDA']['soma'] = []
	SynGroups['NMDA']['lucidum'] = []
	SynGroups['NMDA']['radiatum'] = []
	SynGroups['NMDA']['lacunosumMEC'] = []
	SynGroups['NMDA']['lacunosumLEC'] = []
	
	SynGroups['GABA'] = {}
	SynGroups['GABA']['oriensDistal'] = []
	SynGroups['GABA']['oriensProximal'] = []
	SynGroups['GABA']['oriens'] = []
	SynGroups['GABA']['soma'] = []
	SynGroups['GABA']['lucidum'] = []
	SynGroups['GABA']['radiatum'] = []
	SynGroups['GABA']['lacunosumMEC'] = []
	SynGroups['GABA']['lacunosumLEC'] = []
	
	return SynGroups

# Defines the major axis of the morphology
def getNewAxis():
	new_axis = {}
	new_axis['new_axis'] = [ 0.0534009, 0.99787425, -0.03735406 ]
	return new_axis

# Function to return the nseg resolution
def getNsegRes():
	return 300

# Function to return soma
def getSoma(cell):
	if cell.modeltype == 'Multi':
		return cell.c.soma[0]
	else:
		return cell.soma

# Function to return the "center" of the morphology
# The reference point is set to the somatic location
def getCenter(soma):
	soma.push()
	center = (h.x3d(0),h.y3d(0),h.z3d(0))
	h.pop_section()
	return center

# Function to return to a dendrite lists organized by type (apical or basal)
def getDendTypeList(cell):
	dendTypeList = {}
	dendTypeList['Apical'] = getApicDend(cell)
	dendTypeList['Basal'] = getBasalDend(cell)
	return dendTypeList

# Function to return apical dendrites
def getApicDend(cell):
	return cell.c.apical_dendrite

# Function to return basal dendrites
def getBasalDend(cell):
	return cell.c.dendrite

# Function to return bounds of the layers
def getBounds(maxExtent):
	bounds = {}
	bounds['Apical'] = {}
	bounds['Apical']['soma'] = (0,0)
	bounds['Apical']['lucidum'] = (0,0.1*maxExtent['Apical'])
	bounds['Apical']['radiatum'] = (0.1*maxExtent['Apical'],0.6*maxExtent['Apical'])
	bounds['Apical']['lacunosumMEC'] = (0.6*maxExtent['Apical'],0.8*maxExtent['Apical'])
	bounds['Apical']['lacunosumLEC'] = (0.8*maxExtent['Apical'],maxExtent['Apical'])
	
	bounds['Basal'] = {}
	bounds['Basal']['oriensProximal'] = (0,0.5*maxExtent['Basal'])
	bounds['Basal']['oriensDistal'] = (0.5*maxExtent['Basal'],maxExtent['Basal'])
	
	return bounds

# Function to make the lists containing the locations of the segments
# The list is organized as [ [x], [y], [z] ]
def makeSegLocDict(cell):
	SegLocDict = {}
	SegLocDict['Apical'] = {}
	SegLocDict['Apical']['soma'] = [ [], [], [] ]
	SegLocDict['Apical']['lucidum'] = [ [], [], [] ]
	SegLocDict['Apical']['radiatum'] = [ [], [], [] ]
	SegLocDict['Apical']['lacunosumMEC'] = [ [], [], [] ]
	SegLocDict['Apical']['lacunosumLEC'] = [ [], [], [] ]
	
	SegLocDict['Basal'] = {}
	SegLocDict['Basal']['oriensDistal'] = [ [], [], [] ]
	SegLocDict['Basal']['oriensProximal'] = [ [], [], [] ]

	return SegLocDict

# Function to specify the biophysics of the cell
def getBiophysics(cell):
	ghd = 0.00001 #0.00001
	gna =  0.022 #.022
	gkdr = 0.005 #0.005
	KMULT = 0.02 #0.02
	gc=1.e-5 #1.e-5
	gKc=5.e-5 #5.e-5
	gkm=0.017*1.7 #0.017
	gkd=0.0 # 0.0015 for Ra 150 # 0.0028 for Ra 75 #0.0005 #0.0
	gahp=0.001 #0.0001
	epas = -65 # -48 for Ra 150 # -35.5 for Ra 75
	gcal=gc
	gcan=gc
	gcat=gc
	RA = 140
	Rm = 62996.0
	Cm = 0.72
	PPSpineAdj=1.0
	RCSpineAdj=2.0
	for sec in cell.c.axon:
		sec.insert('na3_CA3')
		sec.insert('kdr_CA3')
		sec.insert('kap_CA3')
		sec.insert('pas')
		sec.gbar_na3_CA3=gna*5
		sec.gkdrbar_kdr_CA3=gkdr
		sec.gkabar_kap_CA3 = KMULT
		sec.sh_kap_CA3=0
		sec.e_pas = epas
		sec.g_pas = 1./Rm
		sec.Ra = 50
		sec.cm = Cm
		#sec.nseg = 9
		sec.ek = -90.0
		sec.ena = 55.0
		sec.sh_na3_CA3 = 24
		sec.sh_kdr_CA3 = 24
		sec.sh_kap_CA3 = 0
		
	for sec in cell.c.dendrite:
		sec.insert('ds_CA3')
		sec.insert('hd_CA3')
		sec.insert('na3_CA3')
		sec.insert('kdr_CA3')
		sec.insert('kap_CA3')
		sec.insert('cacum_CA3')
		sec.depth_cacum_CA3=sec.diam/2
		sec.insert('cal_CA3')
		sec.insert('can_CA3')
		sec.insert('cat_CA3')
		sec.insert('cagk_CA3')
		sec.insert('KahpM95_CA3')
		sec.ehd_hd_CA3=-30.0
		sec.ghdbar_hd_CA3=ghd
		sec.gbar_na3_CA3=gna
		sec.gkdrbar_kdr_CA3=gkdr 
		sec.gkabar_kap_CA3 = KMULT
		sec.gcalbar_cal_CA3=gc
		sec.gcanbar_can_CA3=gc
		sec.gcatbar_cat_CA3=gc
		sec.gbar_cagk_CA3= gKc
		sec.gbar_KahpM95_CA3 = gahp
		sec.insert('pas')
		sec.e_pas = epas
		sec.g_pas = 1./Rm*RCSpineAdj
		sec.Ra = RA
		sec.cm = Cm*RCSpineAdj
		#sec.nseg = 9
		sec.ek = -90.0
		sec.ena = 55.0
		sec.sh_na3_CA3 = 24
		sec.sh_kdr_CA3 = 24
		sec.sh_kap_CA3 = 24
	'''
	for sec in cell.c.apical_dendrite:
		sec.insert('hd_CA3')
		sec.insert('na3_CA3')
		sec.insert('kdr_CA3')
		sec.insert('kap_CA3')
		#sec.ehd_hd=-30.0
		sec.ghdbar_hd_CA3=ghd
		sec.gbar_na3_CA3=gna#*1.25
		sec.gkdrbar_kdr_CA3=gkdr 
		sec.gkabar_kap_CA3 = KMULT
		sec.insert('pas')
		sec.e_pas = epas
		sec.g_pas = 1./Rm
		sec.Ra = RA
		sec.cm = Cm
		#sec.nseg = 9
		sec.ek = -90.0
		sec.ena = 55.0
	'''
	for sec in cell.lucidum:
		sec.insert('hd_CA3')
		sec.insert('na3_CA3')
		sec.insert('kdr_CA3')
		sec.insert('kap_CA3')
		sec.insert('pas')
		sec.ehd_hd_CA3=-30.0
		sec.ghdbar_hd_CA3=ghd
		sec.e_pas = epas
		sec.g_pas = 1./Rm
		sec.Ra = RA
		sec.cm = Cm
		sec.ek = -90.0
		sec.ena = 55.0
		sec.sh_na3_CA3 = 24
		sec.sh_kdr_CA3 = 24
		sec.sh_kap_CA3 = 24
		for seg in cell.lucidum[sec]:
			#sec(seg).gnabar_namr=gna
			sec(seg).gbar_na3_CA3=gna
			sec(seg).gkdrbar_kdr_CA3=gkdr 
			sec(seg).gkabar_kap_CA3 = KMULT
			sec(seg).cm *= RCSpineAdj
			sec(seg).g_pas *= RCSpineAdj

	for sec in cell.radiatum:
		sec.insert('hd_CA3')
		sec.insert('na3_CA3')
		sec.insert('kdr_CA3')
		sec.insert('kap_CA3')
		sec.insert('pas')
		sec.ehd_hd_CA3=-30.0
		sec.ghdbar_hd_CA3=ghd
		sec.e_pas = epas
		sec.g_pas = 1./Rm
		sec.Ra = RA
		sec.cm = Cm
		sec.ek = -90.0
		sec.ena = 55.0
		sec.sh_na3_CA3 = 24
		sec.sh_kdr_CA3 = 24
		sec.sh_kap_CA3 = 24
		for seg in cell.radiatum[sec]:
			#sec(seg).gnabar_namr=gna
			sec(seg).gbar_na3_CA3=gna
			sec(seg).gkdrbar_kdr_CA3=gkdr 
			sec(seg).gkabar_kap_CA3 = KMULT
			sec(seg).cm *= RCSpineAdj
			sec(seg).g_pas *= RCSpineAdj

	for sec in cell.lacunosumMEC:
		sec.insert('na3_CA3')
		sec.insert('hd_CA3')
		sec.insert('kdr_CA3')
		sec.insert('kap_CA3')
		sec.insert('pas')
		sec.ehd_hd_CA3=-30.0
		sec.ghdbar_hd_CA3=ghd
		sec.e_pas = epas
		sec.g_pas = 1./Rm
		sec.Ra = RA
		sec.cm = Cm
		sec.ek = -90.0
		sec.ena = 55.0
		sec.sh_na3_CA3 = 24
		sec.sh_kdr_CA3 = 24
		sec.sh_kap_CA3 = 24
		for seg in cell.lacunosumMEC[sec]:
			#sec(seg).gnabar_namr=gna
			sec(seg).gbar_na3_CA3=gna
			sec(seg).gkdrbar_kdr_CA3=gkdr
			sec(seg).gkabar_kap_CA3 = KMULT
			sec(seg).cm *= PPSpineAdj
			sec(seg).g_pas *= PPSpineAdj
	
	for sec in cell.lacunosumLEC:
		sec.insert('na3_CA3')
		sec.insert('hd_CA3')
		sec.insert('kdr_CA3')
		sec.insert('kap_CA3')
		sec.insert('pas')
		sec.ehd_hd_CA3=-30.0
		sec.ghdbar_hd_CA3=ghd
		sec.e_pas = epas
		sec.g_pas = 1./Rm
		sec.Ra = RA
		sec.cm = Cm
		sec.ek = -90.0
		sec.ena = 55.0
		sec.sh_na3_CA3 = 24
		sec.sh_kdr_CA3 = 24
		sec.sh_kap_CA3 = 24
		for seg in cell.lacunosumLEC[sec]:
			#sec(seg).gnabar_namr=gna
			sec(seg).gbar_na3_CA3=gna
			sec(seg).gkdrbar_kdr_CA3=gkdr
			sec(seg).gkabar_kap_CA3 = KMULT
			sec(seg).cm *= PPSpineAdj
			sec(seg).g_pas *= PPSpineAdj
	
	for sec in cell.c.soma:
		sec.insert('ds_CA3')
		sec.insert('hd_CA3')
		sec.insert('na3_CA3')
		sec.insert('kdr_CA3')
		sec.insert('kap_CA3')
		sec.insert('km_CA3')
		sec.insert('kd_CA3')
		sec.insert('cacum_CA3')
		sec.depth_cacum_CA3=sec.diam/2
		sec.insert('cal_CA3')
		sec.insert('can_CA3')
		sec.insert('cat_CA3')
		sec.insert('cagk_CA3')
		sec.insert('KahpM95_CA3')
		sec.ghdbar_hd_CA3=ghd
		sec.gbar_na3_CA3=gna
		sec.gkdrbar_kdr_CA3=gkdr
		sec.gkabar_kap_CA3 = KMULT
		sec.gbar_km_CA3= gkm
		sec.gkdbar_kd_CA3 = gkd
		sec.ehd_hd_CA3=-30.0
		sec.gcalbar_cal_CA3=gc
		sec.gcanbar_can_CA3=gc
		sec.gcatbar_cat_CA3=gc
		sec.gbar_cagk_CA3= gKc#*1000
		sec.gbar_KahpM95_CA3 = gahp
		sec.insert('pas')
		sec.e_pas = epas
		sec.g_pas = 1./Rm
		sec.Ra = RA
		sec.cm = Cm
		#sec.nseg = 9
		sec.ek = -90.0
		sec.ena = 55.0
		sec.sh_na3_CA3 = 24
		sec.sh_kdr_CA3 = 24
		sec.sh_kap_CA3 = 24

# Function to specify the biophysics of the reduced cell model
def getReducedBiophysics(cell):
    with open('CA3_Bakerb_marascoProp.pickle') as f:
        props = pickle.load(f)
    
    ghd = 0.00001 #0.00001
    gna =  0.022 #.022
    gkdr = 0.005 #0.005
    KMULT = 0.02 #0.02
    gc=1.e-5 #1.e-5
    gKc=5.e-5 #5.e-5
    gkm=0.017*1.7 #0.017
    gkd=0.0 # 0.0015 for Ra 150 # 0.0028 for Ra 75 #0.0005 #0.0
    gahp=0.001 #0.0001
    epas = -65 # -48 for Ra 150 # -35.5 for Ra 75
    gcal=gc
    gcan=gc
    gcat=gc
    RA = 140
    #Rm = 62996.0
    Rm = 71956.90023667467 # New input resistance
    Cm = 0.72
    PPSpineAdj=1.0
    RCSpineAdj=2.0
    
    cell.soma.nseg = 1
    cell.soma.L = 11.218000411987305
    cell.soma.diam = 13.210000038146973
    
    cell.soma.insert('ds_CA3')
    cell.soma.insert('hd_CA3')
    cell.soma.insert('na3_CA3')
    cell.soma.insert('kdr_CA3')
    cell.soma.insert('kap_CA3')
    cell.soma.insert('km_CA3')
    cell.soma.insert('kd_CA3')
    cell.soma.insert('cacum_CA3')
    cell.soma.depth_cacum_CA3=cell.soma.diam/2
    cell.soma.insert('cal_CA3')
    cell.soma.insert('can_CA3')
    cell.soma.insert('cat_CA3')
    cell.soma.insert('cagk_CA3')
    cell.soma.insert('KahpM95_CA3')
    cell.soma.ghdbar_hd_CA3=ghd
    cell.soma.gbar_na3_CA3=gna
    cell.soma.gkdrbar_kdr_CA3=gkdr
    cell.soma.gkabar_kap_CA3 = KMULT
    cell.soma.gbar_km_CA3= gkm
    cell.soma.gkdbar_kd_CA3 = gkd
    cell.soma.ehd_hd_CA3=-30.0
    cell.soma.gcalbar_cal_CA3=gc
    cell.soma.gcanbar_can_CA3=gc
    cell.soma.gcatbar_cat_CA3=gc
    cell.soma.gbar_cagk_CA3= gKc
    cell.soma.gbar_KahpM95_CA3 = gahp
    cell.soma.insert('pas')
    cell.soma.e_pas = epas
    cell.soma.g_pas = 1./Rm
    cell.soma.Ra = RA
    cell.soma.cm = Cm
    cell.soma.ek = -90.0
    cell.soma.ena = 55.0
    cell.soma.sh_na3_CA3 = 24
    cell.soma.sh_kdr_CA3 = 24
    cell.soma.sh_kap_CA3 = 24
    
    cell.axon.nseg = 1
    cell.axon.L = 97.09121768178117
    cell.axon.diam = 1.0293438031901152
    
    cell.axon.insert('na3_CA3')
    cell.axon.insert('kdr_CA3')
    cell.axon.insert('kap_CA3')
    cell.axon.insert('pas')
    cell.axon.gbar_na3_CA3=gna*5
    cell.axon.gkdrbar_kdr_CA3=gkdr
    cell.axon.gkabar_kap_CA3 = KMULT
    cell.axon.sh_kap_CA3=0
    cell.axon.e_pas = epas
    cell.axon.g_pas = 1./Rm
    cell.axon.Ra = 50
    cell.axon.cm = Cm
    cell.axon.ek = -90.0
    cell.axon.ena = 55.0
    cell.axon.sh_na3_CA3 = 24
    cell.axon.sh_kdr_CA3 = 24
    cell.axon.sh_kap_CA3 = 0
    
    cell.oriensProximal.nseg = props['oriensProximal']['nseg']
    cell.oriensProximal.L = props['oriensProximal']['L']
    cell.oriensProximal.diam = props['oriensProximal']['d']
    cell.oriensProximal.Ra = props['oriensProximal']['Ra']
    
    cell.oriensProximal.insert('ds_CA3')
    cell.oriensProximal.insert('hd_CA3')
    cell.oriensProximal.insert('na3_CA3')
    cell.oriensProximal.insert('kdr_CA3')
    cell.oriensProximal.insert('kap_CA3')
    cell.oriensProximal.insert('cacum_CA3')
    cell.oriensProximal.insert('cal_CA3')
    cell.oriensProximal.insert('can_CA3')
    cell.oriensProximal.insert('cat_CA3')
    cell.oriensProximal.insert('cagk_CA3')
    cell.oriensProximal.insert('KahpM95_CA3')
    cell.oriensProximal.insert('pas')
    
    cell.oriensProximal.ghdbar_hd_CA3 = ghd*props['oriensProximal']['fact']
    cell.oriensProximal.gbar_na3_CA3 = gna*props['oriensProximal']['fact']
    cell.oriensProximal.gkdrbar_kdr_CA3 = gkdr*props['oriensProximal']['fact']
    cell.oriensProximal.gkabar_kap_CA3 = KMULT*props['oriensProximal']['fact']
    cell.oriensProximal.gcalbar_cal_CA3 = gc*props['oriensProximal']['fact']
    cell.oriensProximal.gcanbar_can_CA3 = gc*props['oriensProximal']['fact']
    cell.oriensProximal.gcatbar_cat_CA3 = gc*props['oriensProximal']['fact']
    cell.oriensProximal.gbar_cagk_CA3 = gKc*props['oriensProximal']['fact']
    cell.oriensProximal.gbar_KahpM95_CA3 = gahp*props['oriensProximal']['fact']
    cell.oriensProximal.g_pas = 1./Rm*RCSpineAdj*props['oriensProximal']['fact']
    cell.oriensProximal.cm = Cm*RCSpineAdj*props['oriensProximal']['fact']
    
    cell.oriensProximal.depth_cacum_CA3=cell.oriensProximal.diam/2
    cell.oriensProximal.ek = -90.0
    cell.oriensProximal.ena = 55.0
    cell.oriensProximal.ehd_hd_CA3=-30.0
    cell.oriensProximal.e_pas = epas
    cell.oriensProximal.sh_na3_CA3 = 24
    cell.oriensProximal.sh_kdr_CA3 = 24
    cell.oriensProximal.sh_kap_CA3 = 24
    
    cell.oriensDistal.nseg = props['oriensDistal']['nseg']
    cell.oriensDistal.L = props['oriensDistal']['L']
    cell.oriensDistal.diam = props['oriensDistal']['d']
    cell.oriensDistal.Ra = props['oriensDistal']['Ra']
    
    cell.oriensDistal.insert('ds_CA3')
    cell.oriensDistal.insert('hd_CA3')
    cell.oriensDistal.insert('na3_CA3')
    cell.oriensDistal.insert('kdr_CA3')
    cell.oriensDistal.insert('kap_CA3')
    cell.oriensDistal.insert('cacum_CA3')
    cell.oriensDistal.insert('cal_CA3')
    cell.oriensDistal.insert('can_CA3')
    cell.oriensDistal.insert('cat_CA3')
    cell.oriensDistal.insert('cagk_CA3')
    cell.oriensDistal.insert('KahpM95_CA3')
    cell.oriensDistal.insert('pas')
    
    cell.oriensDistal.ghdbar_hd_CA3 = ghd*props['oriensDistal']['fact']
    cell.oriensDistal.gbar_na3_CA3 = gna*props['oriensDistal']['fact']
    cell.oriensDistal.gkdrbar_kdr_CA3 = gkdr*props['oriensDistal']['fact']
    cell.oriensDistal.gkabar_kap_CA3 = KMULT*props['oriensDistal']['fact']
    cell.oriensDistal.gcalbar_cal_CA3 = gc*props['oriensDistal']['fact']
    cell.oriensDistal.gcanbar_can_CA3 = gc*props['oriensDistal']['fact']
    cell.oriensDistal.gcatbar_cat_CA3 = gc*props['oriensDistal']['fact']
    cell.oriensDistal.gbar_cagk_CA3 = gKc*props['oriensDistal']['fact']
    cell.oriensDistal.gbar_KahpM95_CA3 = gahp*props['oriensDistal']['fact']
    cell.oriensDistal.g_pas = 1./Rm*RCSpineAdj*props['oriensDistal']['fact']
    cell.oriensDistal.cm = Cm*RCSpineAdj*props['oriensDistal']['fact']
    
    cell.oriensDistal.depth_cacum_CA3=cell.oriensDistal.diam/2
    cell.oriensDistal.ek = -90.0
    cell.oriensDistal.ena = 55.0
    cell.oriensDistal.ehd_hd_CA3=-30.0
    cell.oriensDistal.e_pas = epas
    cell.oriensDistal.sh_na3_CA3 = 24
    cell.oriensDistal.sh_kdr_CA3 = 24
    cell.oriensDistal.sh_kap_CA3 = 24
    
    cell.lucidum.nseg = props['lucidum']['nseg']
    cell.lucidum.L = props['lucidum']['L']
    cell.lucidum.diam = props['lucidum']['d']
    cell.lucidum.Ra = props['lucidum']['Ra']
    
    cell.lucidum.insert('hd_CA3')
    cell.lucidum.insert('na3_CA3')
    cell.lucidum.insert('kdr_CA3')
    cell.lucidum.insert('kap_CA3')
    cell.lucidum.insert('pas')
    cell.lucidum.ehd_hd_CA3=-30.0
    cell.lucidum.e_pas = epas
    cell.lucidum.ek = -90.0
    cell.lucidum.ena = 55.0
    cell.lucidum.sh_na3_CA3 = 24
    cell.lucidum.sh_kdr_CA3 = 24
    cell.lucidum.sh_kap_CA3 = 24
    
    cell.lucidum.gbar_na3_CA3 = gna*props['lucidum']['fact']
    cell.lucidum.gkdrbar_kdr_CA3 = gkdr*props['lucidum']['fact']
    cell.lucidum.gkabar_kap_CA3 = KMULT*props['lucidum']['fact']
    cell.lucidum.ghdbar_hd_CA3 = ghd*props['lucidum']['fact']
    cell.lucidum.g_pas = 1./Rm*RCSpineAdj*props['lucidum']['fact']
    cell.lucidum.cm = Cm*RCSpineAdj*props['lucidum']['fact']
    
    cell.radiatum.nseg = props['radiatum']['nseg']
    cell.radiatum.L = props['radiatum']['L']
    cell.radiatum.diam = props['radiatum']['d']
    cell.radiatum.Ra = props['radiatum']['Ra']
    
    cell.radiatum.insert('hd_CA3')
    cell.radiatum.insert('na3_CA3')
    cell.radiatum.insert('kdr_CA3')
    cell.radiatum.insert('kap_CA3')
    cell.radiatum.insert('pas')
    cell.radiatum.ehd_hd_CA3=-30.0
    cell.radiatum.e_pas = epas
    cell.radiatum.ek = -90.0
    cell.radiatum.ena = 55.0
    cell.radiatum.sh_na3_CA3 = 24
    cell.radiatum.sh_kdr_CA3 = 24
    cell.radiatum.sh_kap_CA3 = 24
    
    cell.radiatum.gbar_na3_CA3 = gna*props['radiatum']['fact']
    cell.radiatum.gkdrbar_kdr_CA3 = gkdr*props['radiatum']['fact']
    cell.radiatum.gkabar_kap_CA3 = KMULT*props['radiatum']['fact']
    cell.radiatum.ghdbar_hd_CA3 = ghd*props['radiatum']['fact']
    cell.radiatum.g_pas = 1./Rm*RCSpineAdj*props['radiatum']['fact']
    cell.radiatum.cm = Cm*RCSpineAdj*props['radiatum']['fact']
    
    cell.lacunosumMEC.nseg = props['lacunosumMEC']['nseg']
    cell.lacunosumMEC.L = props['lacunosumMEC']['L']
    cell.lacunosumMEC.diam = props['lacunosumMEC']['d']
    cell.lacunosumMEC.Ra = props['lacunosumMEC']['Ra']
    
    cell.lacunosumMEC.insert('hd_CA3')
    cell.lacunosumMEC.insert('na3_CA3')
    cell.lacunosumMEC.insert('kdr_CA3')
    cell.lacunosumMEC.insert('kap_CA3')
    cell.lacunosumMEC.insert('pas')
    cell.lacunosumMEC.ehd_hd_CA3=-30.0
    cell.lacunosumMEC.e_pas = epas
    cell.lacunosumMEC.ek = -90.0
    cell.lacunosumMEC.ena = 55.0
    cell.lacunosumMEC.sh_na3_CA3 = 24
    cell.lacunosumMEC.sh_kdr_CA3 = 24
    cell.lacunosumMEC.sh_kap_CA3 = 24
    
    cell.lacunosumMEC.gbar_na3_CA3 = gna*props['lacunosumMEC']['fact']
    cell.lacunosumMEC.gkdrbar_kdr_CA3 = gkdr*props['lacunosumMEC']['fact']
    cell.lacunosumMEC.gkabar_kap_CA3 = KMULT*props['lacunosumMEC']['fact']
    cell.lacunosumMEC.ghdbar_hd_CA3 = ghd*props['lacunosumMEC']['fact']
    cell.lacunosumMEC.g_pas = 1./Rm*PPSpineAdj*props['lacunosumMEC']['fact']
    cell.lacunosumMEC.cm = Cm*PPSpineAdj*props['lacunosumMEC']['fact']
    
    cell.lacunosumLEC.nseg = props['lacunosumLEC']['nseg']
    cell.lacunosumLEC.L = props['lacunosumLEC']['L']
    cell.lacunosumLEC.diam = props['lacunosumLEC']['d']
    cell.lacunosumLEC.Ra = props['lacunosumLEC']['Ra']
    
    cell.lacunosumLEC.insert('hd_CA3')
    cell.lacunosumLEC.insert('na3_CA3')
    cell.lacunosumLEC.insert('kdr_CA3')
    cell.lacunosumLEC.insert('kap_CA3')
    cell.lacunosumLEC.insert('pas')
    cell.lacunosumLEC.ehd_hd_CA3=-30.0
    cell.lacunosumLEC.e_pas = epas
    cell.lacunosumLEC.ek = -90.0
    cell.lacunosumLEC.ena = 55.0
    cell.lacunosumLEC.sh_na3_CA3 = 24
    cell.lacunosumLEC.sh_kdr_CA3 = 24
    cell.lacunosumLEC.sh_kap_CA3 = 24
    
    cell.lacunosumLEC.gbar_na3_CA3 = gna*props['lacunosumLEC']['fact']
    cell.lacunosumLEC.gkdrbar_kdr_CA3 = gkdr*props['lacunosumLEC']['fact']
    cell.lacunosumLEC.gkabar_kap_CA3 = KMULT*props['lacunosumLEC']['fact']
    cell.lacunosumLEC.ghdbar_hd_CA3 = ghd*props['lacunosumLEC']['fact']
    cell.lacunosumLEC.g_pas = 1./Rm*PPSpineAdj*props['lacunosumLEC']['fact']
    cell.lacunosumLEC.cm = Cm*PPSpineAdj*props['lacunosumLEC']['fact']

# Function to create a synapse at the chosen segment in a section
def createSyn(synvars,sec_choice,seg_choice):
    if synvars['type'] == "E3_NMDA":
        syn = h.E3_NMDA(sec_choice(seg_choice))
    if synvars['type'] == "E2":	
        syn = h.Exp2Syn(sec_choice(seg_choice))
    if synvars['type'] == "E2_Prob":
        syn = h.E2_Prob(sec_choice(seg_choice))
        syn.P = synvars['P']
    if synvars['type'] == "E2_STP_Prob":
        syn = h.E2_STP_Prob(sec_choice(seg_choice))
    if synvars['type'] == "STDPE2":
        syn = h.STDPE2(sec_choice(seg_choice))
    if synvars['type'] == "STDPE2_Clo":
        syn = h.STDPE2_Clo(sec_choice(seg_choice))
    if synvars['type'] == "STDPE2_STP":
        syn = h.STDPE2_STP(sec_choice(seg_choice))
    if synvars['type'] == "STDPE2_Prob":
        syn = h.STDPE2_Prob(sec_choice(seg_choice))
        syn.P = synvars['P']
    #initializes different variables depending on synapse
    if (synvars['type'] == "STDPE2_STP")|(synvars['type'] == "E2_STP_Prob"):
        syn.F1 = synvars['F1']
    if  (synvars['type'] == "STDPE2_Clo" )|( synvars['type'] == "STDPE2_STP")|( synvars['type'] == "STDPE2")| (synvars['type'] == "STDPE2_Prob"):
        syn.wmax = synvars['wmax']
        syn.wmin = synvars['wmin']
        syn.thresh = synvars['thresh']
    if  (synvars['type'] == "E2_Prob" )|( synvars['type'] == "E2_STP_Prob")|(synvars['type'] == "STDPE2_STP") | (synvars['type'] == "STDPE2_Prob"):
        h.use_mcell_ran4(1)
        syn.seed = self.ranGen.randint(1,4.295e9)
    syn.tau1 = 0.5
    syn.tau2 = 0.6
    syn.e = 0        
    return syn

# Function to add synapses to the reduced cell model
def addReducedSynapses(cell):
    for syntype in cell.synGroups:
        synchoice = cell.synvars
        if syntype == 'NMDA':
            synchoice['type'] = 'E3_NMDA'
        
        # oriensDistal
        syn = createSyn(synchoice,cell.oriensDistal,0.5)
        cell.synGroups[syntype]['oriensDistal'].append(syn)
        
        # oriensProximal
        syn = createSyn(synchoice,cell.oriensProximal,0.5)
        cell.synGroups[syntype]['oriensProximal'].append(syn)
        
        # soma
        syn = createSyn(synchoice,cell.soma,0.5)
        cell.synGroups[syntype]['soma'].append(syn)
        
        # lucidum
        syn = createSyn(synchoice,cell.lucidum,0.5)
        cell.synGroups[syntype]['lucidum'].append(syn)
        
        # radiatum
        syn = createSyn(synchoice,cell.radiatum,0.5)
        cell.synGroups[syntype]['radiatum'].append(syn)
        
        # lacunosumMEC
        syn = createSyn(synchoice,cell.lacunosumMEC,0.5)
        cell.synGroups[syntype]['lacunosumMEC'].append(syn)
        
        # lacunosumLEC
        syn = createSyn(synchoice,cell.lacunosumLEC,0.5)
        cell.synGroups[syntype]['lacunosumLEC'].append(syn)

# End of file
