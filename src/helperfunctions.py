import networkx as nx
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


##draw a simple 2D graph using networkx. draw_simple_graph(obj<Graphs>*, int<graph_id>*, str<savefinename>)
def draw_simple_graph(gt, graph_id, filename='plotgraph.png'):
		colorslist =cm.rainbow(np.linspace(0, 1, len(gt.edge_type_id)))
		G = nx.MultiDiGraph()
		pos = nx.spring_layout(G)
		arc_rad = 0.2

		for node_id in range(0,gt.number_of_graph_nodes[graph_id]):
			for node_edge_num in range(0, gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]):
				edge_index = gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
				G.add_edge(str(node_id), str(gt.edge[edge_index][0]), weight=gt.edge[edge_index][1])

		pos = nx.spring_layout(G)
		nx.draw_networkx_nodes(G, pos)
		nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

		legend_elements=[]
		
		for k in range(len(gt.edge_type_id)):
			eset = [(u, v) for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
			elabls = [d for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
			le = 'Unknown'
			for ln in gt.edge_type_id.keys():
				if gt.edge_type_id[ln] == k:
					le = ln
					break
			legend_elements.append(Line2D([0], [0], marker='o', color=colorslist[k], label=le, lw=0))
			
			nx.draw_networkx_edges(G, pos, edgelist=eset, edge_color = colorslist[k], connectionstyle=f'arc3, rad = {arc_rad*k}')
		print(legend_elements)
		plt.title('Graph '+str(graph_id))
		plt.legend(handles=legend_elements, loc='upper left')
		plt.savefig(f"src/outputs/{filename}", dpi=300, bbox_inches='tight')


##print the nodes, seperated by (). show_graph_nodes(obj<Graphs>*, int<graph_id>*)
def show_graph_nodes(gt, graph_id):
		graphstr ='Graph#'+str(graph_id)+':\n'
		for node_id in range(gt.number_of_graph_nodes[graph_id]):
			nodestr='Node#'+str(node_id)+'('
			for (symbol_name, symbol_id) in gt.symbol_id.items():
				match = True
				for k in gt.hypervectors[symbol_id,:]:
					chunk = k // 32
					pos = k % 32

					if (gt.X[gt.node_index[graph_id] + node_id][chunk] & (1 << pos)) == 0:
						match = False

				if match:
					nodestr+= symbol_name+' '
				else:
					nodestr+= '*'+' '
			nodestr+= ')'+' '
			graphstr+= nodestr
		print(graphstr)

##print the edges, grouped by source node. show_graph_edges(obj<Graphs>*, int<graph_id>*)
def show_graph_edges(gt, graph_id):
	graphstr ='Graph#'+str(graph_id)+':\n'
	for node_id in range(0,gt.number_of_graph_nodes[graph_id]):
		for node_edge_num in range(0, gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]):
			edge_index = gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
			edgestr='Edge#'+str(edge_index)
		
			edgestr+= ' SrcNode#'+str(node_id)

			edgestr+= ' DestNode#'+str(gt.edge[edge_index][0])

			edgestr+= ' Type#'+str(gt.edge[edge_index][1])+'\n'
		
			graphstr+= edgestr
		graphstr+= '\n'
	print(graphstr)
