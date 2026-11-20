import igraph, os, sys
import pandas as pd
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score

band_label = pd.read_csv(f"datasets/global/nodes.tsv", sep = '\t')
band_label = band_label[band_label["attribute"].str.startswith("l_")]

band_label = band_label.sort_values(by = "value", ascending = False).groupby(by = "band").agg({"attribute": "first"})


print("""
\\begin{table}
\\centering
\\begin{tabular}{l|r|rrr}
Backbone & Q & Geo & Genre & Label\\\\
\\hline
""")

for bb_method in ("df", "nc", "naive"):
   for bb_decorator in ("top6", "classic"):
      outfile = f"communities_{bb_method}_{bb_decorator}.tsv"
      edges = pd.read_csv(f"edges_{bb_method}_{bb_decorator}.tsv", sep = '\t')
      nodes = sorted(list(set(edges["src"]) | set(edges["trg"])))
      nodemap = {nodes[i]: i for i in range(len(nodes))}
      node_attrs = pd.read_csv(f"datasets/global/nodes_viz.tsv", sep = '\t')
      node_attrs["label"] = node_attrs["band"].map(band_label.to_dict()["attribute"]).fillna("")
      node_attrs["band"] = node_attrs["band"].map(nodemap)
      edges["src"], edges["trg"] = edges["src"].map(nodemap), edges["trg"].map(nodemap)
      g = igraph.Graph()
      g.add_vertices(list(range(len(nodes))))
      g.add_edges(list(edges[["src", "trg"]].itertuples(index = False, name = None)))
      comms = g.community_leiden(objective_function = "modularity")
      node_attrs["comm"] = node_attrs["band"].map({n: i for i in range(len(comms)) for n in comms[i]})
      node_attrs.to_csv(outfile, index = False, sep = "\t")
      ami_geo = adjusted_mutual_info_score(node_attrs["country"], node_attrs["comm"])
      ami_genre = adjusted_mutual_info_score(node_attrs["genre"], node_attrs["comm"])
      ami_label = adjusted_mutual_info_score(node_attrs["label"], node_attrs["comm"])
      print(f"{bb_method} {bb_decorator} & {comms.modularity:.4f} & {ami_geo:.4f} & {ami_genre:.4f} & {ami_label:.4f}\\\\")

print("""
\\end{tabular}
\\caption{The modularity (Q) and adjusted mutual information scores (with geographical origin, main genre, and main label) of the communities for the different backboning approaches.}
\\label{tab:val-comms}
\\end{table}
""")

print(f"Geo-Genre AMI: {adjusted_mutual_info_score(node_attrs['country'], node_attrs['genre']):.4f}")
