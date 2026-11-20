import torch, torch_geometric, os, sys, pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

def process_genres(nodes):
   genres = nodes[nodes["attribute"].str.startswith("g_")]
   genres["value"] = genres["value"].astype(int)
   genres_y = nodes[nodes["attribute"] == "main_style"]
   genre_popularity = genres_y["value"].value_counts() / genres_y["value"].value_counts().sum()
   genres_y["value"] = genres_y["value"].map(lambda x: "other" if x in genre_popularity[genre_popularity < 0.01].index else x)
   genres_y["value"] = genres_y["value"].map({x: list(genres_y["value"].unique()).index(x) for x in genres_y["value"].unique()})
   genres_y = genres_y.set_index("band").loc[range(genres_y.shape[0]), "value"].astype(int)
   return genres, genres_y

def process_years(nodes):
   years = nodes[nodes["attribute"].str.startswith("y_")]
   years["value"] = years["value"].astype(int)
   years_y = nodes[nodes["attribute"] == "avg_year"]
   years_y = years_y.set_index("band").loc[range(years_y.shape[0]), "value"].astype(float)
   return years, years_y

def process_labels(nodes):
   labels = nodes[nodes["attribute"].str.startswith("l_")]
   labels["value"] = labels["value"].astype(int)
   labels_y = labels.groupby(by = "attribute")["value"].sum().sort_values(ascending = False).reset_index()
   biglabels = set(labels_y.iloc[:11]["attribute"]) - set(["l_1818",]) # l_1818 is "not on label" so it shouldn't count
   labels["big"] = labels["attribute"].isin(biglabels)
   labels_y = pd.pivot_table(data = labels.groupby(by = ["band", "big"])["value"].sum().reset_index(), index = "band", columns = "big", values = "value").fillna(0.0)
   labels_y["ratio"] = labels_y[True] / (labels_y[True] + labels_y[False])
   return labels.drop("big", axis = 1), (labels_y["ratio"] > 0.05).astype(int)

def process_origins(nodes):
   origins = nodes.loc[nodes["attribute"] == "origin"]
   countries = origins["value"].drop_duplicates().sort_values().tolist()
   origins_y = origins.set_index("band").loc[range(origins.shape[0]), "value"].map({countries[i]: i for i in range(len(countries))})
   origins_y -= origins_y.min()
   origins["dummy"] = 1
   origins = origins[["band", "value", "dummy"]].rename(columns = {"value": "attribute", "dummy": "value"})
   return origins, origins_y

def process_success(nodemap_reverse):
   df = pd.read_csv(f"datasets/global/nodes_viz.tsv", sep = "\t")[["band", "success"]]
   df["band"] = df["band"].map(nodemap_reverse)
   df = df.dropna().astype(int).sort_values(by = "band").set_index("band")[["success",]]
   df = df.reindex(range(df.shape[0])).fillna(0)
   return np.log10(df["success"] + 1)

def load_network(bb_method, bb_decorator):
   with open(f"aliases_entities", 'rb') as f:
      aliases = pickle.load(f)
   nodes = pd.read_csv(f"datasets/global/nodes.tsv", sep = "\t")
   edges = pd.read_csv(f"edges_{bb_method}_{bb_decorator}.tsv", sep = "\t")
   edges["src"], edges["trg"] = edges["src"].map(lambda x: aliases[x] if x in aliases else x), edges["trg"].map(lambda x: aliases[x] if x in aliases else x)
   G = nx.from_pandas_edgelist(edges, source = "src", target = "trg", edge_attr = True)
   G = G.subgraph(max(nx.connected_components(G), key = len))
   nodemap = sorted(list(G.nodes))
   nodemap_reverse = {int(nodemap[i]): i for i in range(len(nodemap))}
   nodes["band"] = nodes["band"].map(nodemap_reverse)
   edges["src"], edges["trg"] = edges["src"].map(nodemap_reverse), edges["trg"].map(nodemap_reverse)
   edges = edges.dropna()
   edges[["src", "trg"]] = edges[["src", "trg"]].astype(int)
   return nodes.dropna().sort_values(by = "band"), edges, nodemap_reverse

def make_tensor(tensor_type, country, edges, genres, years, labels, origins, y):
   tensor = torch_geometric.data.Data(
      edge_index = torch.tensor(np.array([pd.concat([edges["src"], edges["trg"]]).values, pd.concat([edges["trg"], edges["src"]]).values]), dtype = torch.int),
      edge_attr = torch.tensor(pd.concat([edges[["score",]], edges[["score",]]]).values, dtype = torch.float32),
   )
   if tensor_type == "country":
      x = pd.concat([genres, years, labels])
      tensor.y = torch.tensor(y.values.astype(int), dtype = torch.long)
      tensor.train_mask = torch.tensor(y.index.isin(train_test_split(y.index, test_size = 0.2, stratify = y)[0]), dtype = torch.bool)
   elif tensor_type == "genre":
      x = pd.concat([years, labels, origins]) if country == "global" else pd.concat([years, labels])
      tensor.y = torch.tensor(y.values.astype(int), dtype = torch.long)
      tensor.train_mask = torch.tensor(y.index.isin(train_test_split(y.index, test_size = 0.2, stratify = y)[0]), dtype = torch.bool)
   elif tensor_type == "year":
      x = pd.concat([genres, labels, origins]) if country == "global" else pd.concat([genres, labels])
      tensor.y = torch.tensor(y.values, dtype = torch.float32)
      stratification = y // 50
      breakpoint()
      stratification[stratification == 37] = 38
      tensor.train_mask = torch.tensor(y.index.isin(train_test_split(y.index, test_size = 0.2, stratify = stratification)[0]), dtype = torch.bool)
   elif tensor_type == "label":
      x = pd.concat([genres, years, origins]) if country == "global" else pd.concat([genres, years])
      tensor.y = torch.tensor(y.values, dtype = torch.long)
      tensor.train_mask = torch.tensor(y.index.isin(train_test_split(y.index, test_size = 0.2, stratify = y)[0]), dtype = torch.bool)
   elif tensor_type == "success":
      x = pd.concat([genres, labels, years, origins]) if country == "global" else pd.concat([genres, years, labels])
      y = (y//1).map(lambda x: 6 if x == 7 else x).astype(int)
      tensor.y = torch.tensor(y, dtype = torch.long)
      tensor.train_mask = torch.tensor(y.index.isin(train_test_split(y.index, test_size = 0.2, stratify = y)[0]), dtype = torch.bool)
   x_attrs = x["attribute"].drop_duplicates().sort_values().values
   x_attrs = {x_attrs[i]: i for i in range(x_attrs.shape[0])}
   x["attribute"] = x["attribute"].map(x_attrs)
   size = (x["band"].max() + 1, x["attribute"].max() + 1)
   x_sparse = coo_matrix((x["value"], (x["band"], x["attribute"])), shape = size)
   tensor.x = torch.sparse_coo_tensor(np.vstack((x_sparse.row, x_sparse.col)), x_sparse.data, dtype = torch.float32, size = size).to_sparse_csr()
   x_sparse_train = x_sparse.tocsr()[tensor.train_mask.numpy()].tocoo()
   x_sparse_test = x_sparse.tocsr()[~tensor.train_mask.numpy()].tocoo()
   tensor.x_train = torch.sparse_coo_tensor(np.vstack((x_sparse_train.row, x_sparse_train.col)), x_sparse_train.data, dtype = torch.float32, size = (tensor.train_mask.sum(), size[1])).to_sparse_csr()
   tensor.x_test = torch.sparse_coo_tensor(np.vstack((x_sparse_test.row, x_sparse_test.col)), x_sparse_test.data, dtype = torch.float32, size = ((~tensor.train_mask).sum(), size[1])).to_sparse_csr()
   torch.save(tensor, f"tensor_{bb_method}_{bb_decorator}_{tensor_type}.pt")
   del tensor

def build_tensor(bb_method, bb_decorator):
   sys.stderr.write(f"{bb_method} {bb_decorator}...\n")
   nodes, edges, nodemap_reverse = load_network(bb_method, bb_decorator)
   success_y = process_success(nodemap_reverse)
   genres, genres_y = process_genres(nodes)
   years, years_y = process_years(nodes)
   labels, labels_y = process_labels(nodes)
   origins, origins_y = process_origins(nodes)
   make_tensor("country", "global", edges, genres, years, labels, origins, origins_y)
   make_tensor("genre", "global", edges, genres, years, labels, origins, genres_y)
   make_tensor("year", "global", edges, genres, years, labels, origins, years_y)
   make_tensor("label", "global", edges, genres, years, labels, origins, labels_y)
   make_tensor("success", "global", edges, genres, years, labels, origins, success_y)

for bb_method in ("df", "nc", "naive"):
   for bb_decorator in ("top6", "classic"):
      build_tensor(bb_method, bb_decorator)
