import os, sys
# path to circuit-tracer
module_dir = os.path.abspath('/export/home/rmitsuhashi/circuit-tracer')
sys.path.insert(0, module_dir)
module_dir2 = os.path.join(module_dir, 'demos')
sys.path.insert(0, module_dir2)

# import so that you do not need to provide the path all the time
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import Graph, prune_graph
from graph_visualization import Feature, Supernode, InterventionGraph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder


from collections import namedtuple



Intervention = namedtuple('Intervention', ['supernode', 'scaling_factor'])
