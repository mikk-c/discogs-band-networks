import igraph, os, sys
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

corr_method_decorator = defaultdict(lambda : defaultdict(list))

print("""
\\begin{table}
\\centering
\\begin{tabular}{l|rrr}
Backbone & Degree & Closeness & Betweenness\\\\
\\hline
""")

for bb_method in ("df", "nc", "naive"):
   for bb_decorator in ("top6", "classic"):
      outfile = f"centralities_{bb_method}_{bb_decorator}.tsv"
      if not os.path.isfile(outfile):
         edges = pd.read_csv(f"edges_{bb_method}_{bb_decorator}.tsv", sep = '\t')
         nodes = sorted(list(set(edges["src"]) | set(edges["trg"])))
         nodemap = {nodes[i]: i for i in range(len(nodes))}
         node_attrs = pd.read_csv(f"datasets/global/nodes_viz.tsv", sep = '\t')
         node_attrs["band"] = node_attrs["band"].map(nodemap)
         edges["src"], edges["trg"] = edges["src"].map(nodemap), edges["trg"].map(nodemap)
         g = igraph.Graph()
         g.add_vertices(list(range(len(nodes))))
         g.add_edges(list(edges[["src", "trg"]].itertuples(index = False, name = None)))
         node_attrs["degree"] = g.degree()
         node_attrs["betweenness"] = g.betweenness()
         node_attrs["closeness"] = g.closeness()
         node_attrs.to_csv(outfile, index = False, sep = "\t")
      else:
         node_attrs = pd.read_csv(outfile, sep = '\t')
      pear_deg = pearsonr(node_attrs['degree'], node_attrs['success'])
      pear_clo = pearsonr(node_attrs['closeness'], node_attrs['success'])
      pear_bet = pearsonr(node_attrs['betweenness'], node_attrs['success'])
      spea_deg = spearmanr(node_attrs['degree'], node_attrs['success'])
      spea_clo = spearmanr(node_attrs['closeness'], node_attrs['success'])
      spea_bet = spearmanr(node_attrs['betweenness'], node_attrs['success'])
      print(f"{bb_method} {bb_decorator} & ${spea_deg[0]:.4f}^{{{'***' if spea_deg[1] < 0.001 else ''}}}$  & ${spea_clo[0]:.4f}^{{{'***' if spea_clo[1] < 0.001 else ''}}}$  & ${spea_bet[0]:.4f}^{{{'***' if spea_bet[1] < 0.001 else ''}}}$\\\\")

print("""
\\end{tabular}
\\caption{The Spearman correlation coefficients between success and various centrality measures for the different backboning approaches ($^{***}$ indicates a p-value below 0.001).}
\\label{tab:val-corr}
\\end{table}
""")
