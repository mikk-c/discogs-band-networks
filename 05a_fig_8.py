import pandas as pd

edges = {}

for bb_method in ("df", "nc", "naive"):
   for bb_decorator in ("top6", "classic"):
      edges[f"{bb_method}_{bb_decorator}"] = set(pd.read_csv(f"edges_{bb_method}_{bb_decorator}.tsv", sep = '\t')[["src", "trg"]].itertuples(index = False, name = None))

df = []

for bb_method1 in edges:
   for bb_method2 in edges:
      df.append((bb_method1, bb_method2, len(edges[bb_method1] & edges[bb_method2]) / len(edges[bb_method1] | edges[bb_method2])))

df = pd.DataFrame(data = df, columns = ("src", "trg", "jaccard"))
df = df.pivot_table(index = "src", columns = "trg", values = "jaccard")

df = df.loc[["nc_classic", "nc_top6", "df_top6", "naive_top6", "df_classic", "naive_classic"], ["nc_classic", "nc_top6", "df_top6", "naive_top6", "df_classic", "naive_classic"]]
df.to_csv("fig_8.tsv", index = False, sep = "\t", header = False)

