import igraph, os, sys
import pandas as pd
from collections import defaultdict

print("""
\\begin{table}
\\centering
\\begin{tabular}{l|r}
Backbone & Avg Year\\\\
\\hline
""")

for bb_method in ("df", "nc", "naive"):
   for bb_decorator in ("top6", "classic"):
      edges = pd.read_csv(f"edges_{bb_method}_{bb_decorator}.tsv", sep = '\t')
      nodes = sorted(list(set(edges["src"]) | set(edges["trg"])))
      nodemap = {nodes[i]: i for i in range(len(nodes))}
      node_attrs = pd.read_csv(f"datasets/global/nodes_viz.tsv", sep = '\t')
      node_attrs["band"] = node_attrs["band"].map(nodemap)
      edges["src"], edges["trg"] = edges["src"].map(nodemap), edges["trg"].map(nodemap)
      g = igraph.Graph()
      g.add_vertices(list(range(len(nodes))))
      g.add_edges(list(edges[["src", "trg"]].itertuples(index = False, name = None)))
      g.vs["year"] = node_attrs.sort_values(by = "band")["year"]
      ass_year = g.assortativity(g.vs["year"])
      print(f"{bb_method} {bb_decorator} & {ass_year:.4f}\\\\")

print("""
\\end{tabular}
\\caption{The assortativity scores with the average year of relevance for the different backboning approaches.}
\\label{tab:val-assortativity}
\\end{table}
""")
