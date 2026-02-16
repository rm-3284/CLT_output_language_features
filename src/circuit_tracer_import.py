# Import from circuit-tracer package (installed from GitHub)
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import Graph, prune_graph
from circuit_tracer.demos.graph_visualization import Feature, Supernode, InterventionGraph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder

from collections import namedtuple



Intervention = namedtuple('Intervention', ['supernode', 'scaling_factor'])
