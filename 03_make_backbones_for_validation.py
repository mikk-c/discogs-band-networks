import sys
sys.stderr.write("Setting up...\n")
import os, pickle, glob
import pandas as pd
import networkx as nx
import backboning as bb
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict

# This script is a copy of a subpart of script 02, which only creates the edgelist, leaving the nodelist intact
# This is because we just generate different edgelists with different backboning techniques
# And therefore the node attributes do not change from script 02 and can be reused

def reconnect_singletons(df, df_bb, nodes):
   additional_edges = []
   missing_nodes = nodes - (set(df_bb["src"]) | set(df_bb["trg"]))
   while len(missing_nodes) > 0:
      edges_added = df.loc[list(set([x[1] for x in df[df["src"].isin(missing_nodes) | df["trg"].isin(missing_nodes)].groupby(by = "src")["score"].nlargest(1).index]))]
      if edges_added.shape[0] == 0:
         break
      df_bb = pd.concat([df_bb, edges_added])
      missing_nodes -= (set(edges_added["src"]) | set(edges_added["trg"]))
   return df_bb

def reconnect_components(df, df_bb):
   G = nx.from_pandas_edgelist(df_bb, source = "src", target = "trg")
   ccs = list(nx.connected_components(G))
   ccs = {n: i for i in range(len(ccs)) for n in ccs[i]}
   df["src_comp"] = df["src"].map(ccs)
   df["trg_comp"] = df["trg"].map(ccs)
   new_edge_candidates = df.loc[df["src_comp"] != df["trg_comp"], "score"]
   while df[["src_comp", "trg_comp"]].value_counts().size > 1 and new_edge_candidates.shape[0] > 0:
      new_edge = df.loc[new_edge_candidates.idxmax()]
      df_bb = pd.concat([df_bb, pd.DataFrame(new_edge[["src", "trg", "nij", "score"]]).T])
      new_comp = new_edge[["src_comp", "trg_comp"]].min()
      old_comp = new_edge[["src_comp", "trg_comp"]].max()
      df.loc[df["src_comp"] == old_comp, "src_comp"] = new_comp
      df.loc[df["trg_comp"] == old_comp, "trg_comp"] = new_comp
      new_edge_candidates = df.loc[df["src_comp"] != df["trg_comp"], "score"]
   return df_bb

def sparse_histogram_intersection(A):
   Ac = A.tocsc()
   n_rows = A.shape[0]
   result = defaultdict(float)
   for col in range(Ac.shape[1]):
      start, end = Ac.indptr[col], Ac.indptr[col + 1]
      rows = Ac.indices[start:end]
      vals = Ac.data[start:end]
      m = len(rows)
      for i in range(m):
         r1, v1 = rows[i], vals[i]
         for j in range(i + 1, m):
            r2, v2 = rows[j], vals[j]
            result[(r1, r2)] += min(v1, v2)
   return pd.DataFrame(data = [(x, y, result[x,y]) for x, y in result], columns = ("src", "trg", "nij"))

def project(df):
   sys.stderr.write("Projecting bipartite...\n")
   df = df.groupby(by = ["band", "artist"]).size().reset_index()
   rowmap = df["band"].drop_duplicates().sort_values().values
   rowmap = {i: int(rowmap[i]) for i in range(rowmap.shape[0])}
   rowmap_inv = {rowmap[i]: i for i in range(len(rowmap))}
   colmap = df["artist"].drop_duplicates().sort_values().values
   colmap_inv = {colmap[i]: i for i in range(len(colmap))}
   df["band"] = df["band"].map(rowmap_inv)
   df["artist"] = df["artist"].map(colmap_inv)
   df = csr_matrix(coo_matrix((df[0], (df["band"], df["artist"]))))
   df = sparse_histogram_intersection(df)
   df = pd.concat([df, df.rename(columns = {"src": "trg", "trg": "src"})])
   df["src"] = df["src"].map(rowmap)
   df["trg"] = df["trg"].map(rowmap)
   return df

def write_bb(df, bb_method, bb_decorator):
   sys.stderr.write(f"Backboning {bb_method} {bb_decorator}...\n")
   if bb_method == "df":
      df_bb = bb.disparity_filter(df)
   elif bb_method == "nc":
      df_bb = bb.noise_corrected(df)
   elif bb_method == "naive":
      df_bb = bb.naive(df)
   if bb_decorator == "top6":
      edge_indexes = set([x[1] for x in df_bb.groupby(by = "src")["score"].nlargest(6).index])
      df_bb_filtered = df_bb.loc[list(edge_indexes)]
   elif bb_decorator == "classic":
      if bb_method == "df":
         df_bb_filtered = df_bb[df_bb["score"] > 0.999995]
      elif bb_method == "nc":
         df_bb_filtered = bb.thresholding(df_bb, 55000)
      elif bb_method == "naive":
         df_bb_filtered = df_bb[df_bb["score"] > 170]
   df_bb_filtered["src"], df_bb_filtered["trg"] = df_bb_filtered[["src", "trg"]].min(axis = 1), df_bb_filtered[["src", "trg"]].max(axis = 1)
   df_bb_filtered = df_bb_filtered.drop_duplicates(subset = ["src", "trg"])
   df_bb = df_bb[df_bb["src"] < df_bb["trg"]]
   df_bb_filtered = reconnect_singletons(df_bb, df_bb_filtered, set(df_bb["src"]) | set(df_bb["trg"]))
   df_bb_filtered = reconnect_components(df_bb, df_bb_filtered)
   df_bb_filtered[["src", "trg", "nij"]] = df_bb_filtered[["src", "trg", "nij"]].astype(int)
   nodeset = nx.from_pandas_edgelist(df_bb_filtered, source = "src", target = "trg")
   nodeset = max(nx.connected_components(nodeset), key = len)
   df_bb_filtered = df_bb_filtered[df_bb_filtered["src"].isin(nodeset) & df_bb_filtered["trg"].isin(nodeset)]
   sys.stderr.write("Writing edgelists...\n")
   df_bb_filtered.to_csv(f"edges_{bb_method}_{bb_decorator}.tsv", sep = "\t", index = False)

def select_country(country):
   band_country = []
   os.makedirs(f"datasets/{country}", exist_ok = True)
   sys.stderr.write("Loading nodesets...\n")
   for txt in glob.glob("data_selection/preliminary_nodelist*"):
      df = pd.read_csv(txt, sep = "\t")
      df = df.loc[df["in"] == 1, ["canonical_id",]]
      df["country"] = txt.split("_")[-1].split(".")[0]
      band_country.append(df)
   df = pd.read_csv(f"data_selection/excluded_bands_by_success.tsv", sep = "\t")
   df = df.loc[(df["in"] != "0") & (df["in"] != "-1") & ~df["in"].isnull(), ["band", "in"]].rename(columns = {"band": "canonical_id", "in": "country"})
   band_country.append(df)
   band_country = pd.concat(band_country)
   with open(f"aliases_entities", 'rb') as f:
      aliases = pickle.load(f)
   band_country["canonical_id"] = band_country["canonical_id"].map(lambda x: aliases[x] if x in aliases else x)
   band_country = band_country.drop_duplicates(subset = ["canonical_id",])
   return band_country

sys.stderr.write("Loading bipartite network...\n")
bipartite = pd.read_csv("release_band_artist_role.tsv", sep = "\t")

band_country = select_country("global")
bipartite = bipartite[bipartite["band"].isin(set(band_country["canonical_id"]))]
bipartite = project(bipartite)

write_bb(bipartite, "df", "top6")
write_bb(bipartite, "nc", "top6")
write_bb(bipartite, "naive", "top6")
write_bb(bipartite, "df", "classic")
write_bb(bipartite, "nc", "classic")
write_bb(bipartite, "naive", "classic")

