import sys
sys.stderr.write("Setting up...\n")
import os, pickle, glob
import pandas as pd
import networkx as nx
import backboning as bb
from scipy.sparse import coo_matrix, csc_matrix
from collections import defaultdict

# Once the backboning step has been done, there could be nodes with no edges left
# This function will recover them and re-add them to the backbone by finding their most significant edge
def reconnect_singletons(df, df_bb, nodes):
   additional_edges = []
   # nodes is the original set of all nodes before the backbone
   # We subtract from it the set of nodes with at least one edge in the backbone
   # Finding all the original nodes without an edge
   missing_nodes = nodes - (set(df_bb["src"]) | set(df_bb["trg"]))
   # We need to keep going as long as we have nodes that should be recovered
   while len(missing_nodes) > 0:
      # We find all edges attached to a missing nodes and, for each source node, we pick the most significant
      edges_added = df.loc[list(set([x[1] for x in df[df["src"].isin(missing_nodes) | df["trg"].isin(missing_nodes)].groupby(by = "src")["score"].nlargest(1).index]))]
      # If there were no further edge candidates we break the loop
      # This can happen if there was at least one node that was also isolated before taking the backbone
      if edges_added.shape[0] == 0:
         break
      # We add the edges to the backbone
      df_bb = pd.concat([df_bb, edges_added])
      # We remove from the missing node set all nodes that got an edge
      # There could still be missing nodes in case two singletons wanted to connect to the same node, in which case only the most significant was picked
      missing_nodes -= (set(edges_added["src"]) | set(edges_added["trg"]))
   return df_bb

# We need backbones to be a single connected component
# This function finds the most significant edge between components and adds them to the network until it is a single connected component
def reconnect_components(df, df_bb):
   # We transform the pandas dataframe into a networkx object
   G = nx.from_pandas_edgelist(df_bb, source = "src", target = "trg")
   # We find the connected components of the graph
   ccs = list(nx.connected_components(G))
   # We give each component a numeric id, mapping the relationship node -> component id
   ccs = {n: i for i in range(len(ccs)) for n in ccs[i]}
   # We add the component id as a column of the dataframe for both endpoints of an edge
   df["src_comp"] = df["src"].map(ccs)
   df["trg_comp"] = df["trg"].map(ccs)
   # The edges we potentiall add are only the ones between components with different ids
   # I.e. different components
   new_edge_candidates = df.loc[df["src_comp"] != df["trg_comp"], "score"]
   # We keep going until we either have one single component or we ran out of edges we can add
   # The latter is triggered if the network before backboning already had multiple components
   while df[["src_comp", "trg_comp"]].value_counts().size > 1 and new_edge_candidates.shape[0] > 0:
      # We find the most significant edge among the ones we can add
      new_edge = df.loc[new_edge_candidates.idxmax()]
      # We add the edge to the backbone
      df_bb = pd.concat([df_bb, pd.DataFrame(new_edge[["src", "trg", "nij", "score"]]).T])
      # Now the two components are connected, so we need to take one of the two components ids and overwrite it with the other
      # Because all nodes that were the two separate components are now in the same
      new_comp = new_edge[["src_comp", "trg_comp"]].min()
      old_comp = new_edge[["src_comp", "trg_comp"]].max()
      df.loc[df["src_comp"] == old_comp, "src_comp"] = new_comp
      df.loc[df["trg_comp"] == old_comp, "trg_comp"] = new_comp
      # Now that we have joined the two components, we throw out from the candidate edges the one that were connecting them
      new_edge_candidates = df.loc[df["src_comp"] != df["trg_comp"], "score"]
   return df_bb

# This is the function that calculates the historgram intersection
# I.e. how we determine the edge weights for the unipartite projection of the original bipartite network
def sparse_histogram_intersection(A):
   result = defaultdict(float)
   # We loop over every column (artist) of the bipartite adjacency matrix
   for col in range(A.shape[1]):
      # We extract the data in the column from the CSC sparse format
      start, end = A.indptr[col], A.indptr[col + 1]
      rows = A.indices[start:end]
      vals = A.data[start:end]
      m = len(rows)
      # We now loop over every non-zero row pair (pair of bands for which the artist played) in that column
      for i in range(m):
         r1, v1 = rows[i], vals[i]
         for j in range(i + 1, m):
            r2, v2 = rows[j], vals[j]
            # The edge weight between bands r1 and r2 is increased by the minimum of their values
            result[(r1, r2)] += min(v1, v2)
   return pd.DataFrame(data = [(x, y, result[x,y]) for x, y in result], columns = ("src", "trg", "nij"))

def write_cumulative_year(bipartite, year):
   # We want to create a network up until a given year, so we filter out all releases that came out after that year
   df = bipartite[bipartite["year"] <= year].groupby(by = ["band", "artist"]).size().reset_index()
   # Because we might be looking at a country which didn't have releases before a given year, the resulting dataframe might be empty
   if df.shape[0] == 0:
      return set()
   # We now create consecutive numeric node ids stating from 0 for both bands and artists
   # This is necessary to transform the pandas DataFrame into a sparse network
   # We start by creating the map from Discogs ID to numeric id
   rowmap = df["band"].drop_duplicates().sort_values().values
   rowmap = {i: int(rowmap[i]) for i in range(rowmap.shape[0])}
   colmap = df["artist"].drop_duplicates().sort_values().values
   # And then we invert the map so we can transform each Discogs id into a numeric id
   rowmap_inv = {rowmap[i]: i for i in range(len(rowmap))}
   colmap_inv = {colmap[i]: i for i in range(len(colmap))}
   # We apply the inverse id map
   df["band"] = df["band"].map(rowmap_inv)
   df["artist"] = df["artist"].map(colmap_inv)
   # With the inverse id map applied, it is now trivial to build a sparse scipy matrix
   df = csc_matrix(coo_matrix((df[0], (df["band"], df["artist"]))))
   # We project the bipartite network using our histogram intersection approach
   # The result is a pandas dataframe with the weighted unipartite projection
   df = sparse_histogram_intersection(df)
   # It's possible that the projection is empty
   if df.shape[0] == 0:
      return set()
   # We now duplicate each edge, so that the backboning will estimate the significance of an edge for both direction
   # The disparity filter can give different significance scores for the u,v direction than the v,u direction
   # Even if the network is undirected
   df = pd.concat([df, df.rename(columns = {"src": "trg", "trg": "src"})])
   # We undo the node id mapping to go back to Discogs ids
   df["src"] = df["src"].map(rowmap)
   df["trg"] = df["trg"].map(rowmap)
   # We apply the backboning, estimating the significance of each edge
   df = bb.disparity_filter(df)
   # For each node, we pick the 6 most significant edges from their perspective
   # If a node has fewer than 6 edges, this quietly simply returns all of them
   edge_indexes = set([x[1] for x in df.groupby(by = "src")["score"].nlargest(6).index])
   # Before we just selected the indexes in the dataframe of the significant edges, now we filter the dataframe
   df_bb = df.loc[list(edge_indexes)]
   # We do not need to keep each edge copied in both directions, so we drop the duplicates
   # By convention, we make sure each edge has as "src" the node with the lowest id of the two
   df_bb["src"], df_bb["trg"] = df_bb[["src", "trg"]].min(axis = 1), df_bb[["src", "trg"]].max(axis = 1)
   df_bb = df_bb.drop_duplicates(subset = ["src", "trg"])
   df = df[df["src"] < df["trg"]]
   # Because we want to recover singletons and have a single connecgted components, we need to re-add a handful of edges
   df_bb = reconnect_singletons(df, df_bb, set(df["src"]) | set(df["trg"]))
   df_bb = reconnect_components(df, df_bb)
   df_bb[["src", "trg", "nij"]] = df_bb[["src", "trg", "nij"]].astype(int)
   # Even if we tried to connect the various components, we might still have more than one
   # Because the original network could have been disconnected even before backboning
   # If that's the case, we will only work with the largest connected component
   nodeset = nx.from_pandas_edgelist(df_bb, source = "src", target = "trg")
   nodeset = max(nx.connected_components(nodeset), key = len)
   df = df[df["src"].isin(nodeset) & df["trg"].isin(nodeset)]
   df_bb = df_bb[df_bb["src"].isin(nodeset) & df_bb["trg"].isin(nodeset)]
   # We save the edge list to file
   df_bb.to_csv(f"datasets/{country}/edges_{year}.tsv", sep = "\t", index = False)
   # We return the set of nodes in the backbone because we will need it to know for which bands we need the node attributes
   return nodeset

def select_country(country):
   band_country = []
   # We create the folder that will host the country's output files, if it doesn't exist already
   os.makedirs(f"datasets/{country}", exist_ok = True)
   sys.stderr.write("Loading nodesets...\n")
   if country == "global":
      # If we want to make the global network, we need to get the band ids from all regions we preprocessed
      for txt in glob.glob("data_selection/preliminary_nodelist*"):
         df = pd.read_csv(txt, sep = "\t")
         # In each file, the band has a 1 in the "in" column if we judged it to be part of that region
         # We then just need its id
         df = df.loc[df["in"] == 1, ["canonical_id",]]
         # We figure out which country we just loaded by the filename, handling New Zealand's edge case
         if "zealand" in txt:
            df["country"] = "_".join(txt.split("_")[-2:]).split(".")[0]
         else:
            df["country"] = txt.split("_")[-1].split(".")[0]
         band_country.append(df)
      # The files we parsed so far only include bands according to how many releases they published
      # This second archive is the result of the second pass, when we select bands by their success
      df = pd.read_csv(f"data_selection/excluded_bands_by_success.tsv", sep = "\t")
      # We need to filter differently, because bands in this file could be in any region
      # So we need to explicitly drop 0 (= band is successful enough but comes from a region we don't study)
      # And -1 (= band is from a region we study but it is not a music band)
      # And use the content of the "in" column to figure out from which region the band hails, if we should include it
      df = df.loc[(df["in"] != "0") & (df["in"] != "-1") & ~df["in"].isnull(), ["band", "in"]].rename(columns = {"band": "canonical_id", "in": "country"})
      band_country.append(df)
   else:
      # If we're here, we're focusing on a single country, we start by reading its archive based on release count
      band_country.append(pd.read_csv(f"data_selection/preliminary_nodelist_{country}.tsv", sep = "\t"))
      # We only need the included bands, and we only need their ids
      band_country[0] = band_country[0].loc[band_country[0]["in"] == 1, ["canonical_id",]]
      band_country[0]["country"] = country
      # We then need to check the archive based on bands' success and take only those from the current country (looking at the "in" column)
      df = pd.read_csv(f"data_selection/excluded_bands_by_success.tsv", sep = "\t")
      df = df.loc[df["in"] == country, ["band", "in"]].rename(columns = {"band": "canonical_id", "in": "country"})
      band_country.append(df)
   # We concatenate all dataframes we have so far, which all contain the band ids we want and their geographical origin
   band_country = pd.concat(band_country)
   # We load the canonical ids of bands
   with open(f"aliases_entities", 'rb') as f:
      aliases = pickle.load(f)
   # And we make sure each band we include is included only using their canonical id
   band_country["canonical_id"] = band_country["canonical_id"].map(lambda x: aliases[x] if x in aliases else x)
   # If by any chance two bands were the same, we keep only one of them
   band_country = band_country.drop_duplicates(subset = ["canonical_id",])
   return band_country

def write_node_attributes(nodeset, band_country, bipartite_country, release_attrs):
   sys.stderr.write("Loading genres...\n")
   # This dataframe will connect each release with its music genre(s)
   genres = pd.read_csv("release_genres.tsv", sep = "\t")
   # By merging it with the bipartite network of the region, we only keep the valid releases from bands of that region
   genres = genres.merge(bipartite_country[["release", "band"]].drop_duplicates(), on = "release")
   # We can now count how many releases a band has with a given genre
   genres = genres.groupby(by = ["band", "genre"]).size().reset_index()
   # The main genre of the band is the one with the highest count
   main_genre = genres.loc[[x[1] for x in genres.groupby(by = "band")[0].nlargest(1).index]].drop(0, axis = 1)
   sys.stderr.write("Loading styles...\n")
   # This dataframe will connect each release with its music style(s)
   styles = pd.read_csv("release_styles.tsv", sep = "\t")
   # By merging it with the bipartite network of the region, we only keep the valid releases from bands of that region
   styles = styles.merge(bipartite_country[["release", "band"]].drop_duplicates(), on = "release")
   # We can now count how many releases a band has with a given style
   styles = styles.groupby(by = ["band", "style"]).size().reset_index()
   sys.stderr.write("Loading labels...\n")
   # This dataframe will connect each release with its publishing label(s)
   labels = pd.read_csv("release_labels.tsv", sep = "\t").drop_duplicates()
   # By merging it with the bipartite network of the region, we only keep the valid releases from bands of that region
   labels = labels.merge(bipartite_country[["release", "band"]].drop_duplicates(), on = "release")
   # We can now count how many releases a band has with a given label
   labels = labels.groupby(by = ["band", "label"]).size().reset_index()
   sys.stderr.write("Loading years...\n")
   # We filter the dataframe with the release's year and success measures to keep the valid releases from bands of that region
   release_attrs_country = release_attrs.merge(bipartite_country[["release", "band"]].drop_duplicates(), on = "release")
   # We can calculate the average year of release of a band
   avg_year = release_attrs_country.groupby(by = "band")["year"].mean().reset_index()
   sys.stderr.write("Generating node attributes...\n")
   # We make sure we only look at bands that made it in the largest connected component of the backbone
   band_country = band_country[band_country["canonical_id"].isin(nodeset)]
   bands = pd.read_csv("band_attributes.tsv", sep = "\t")
   bands = bands[bands["band"].isin(nodeset)]
   # We now add to the band attribute dataframe all summary attributes we generated: main genre, average year, names, and success
   bands = bands.merge(main_genre, on = "band")
   bands = bands.merge(avg_year, on = "band")
   bands = bands.merge(band_country, left_on = "band", right_on = "canonical_id").drop("canonical_id", axis = 1)
   bands = bands.rename(columns = {"name": "band_name"})
   # Since this is few attributes, it'll be saved in dense format
   bands.to_csv(f"datasets/{country}/nodes_viz.tsv", index = False, sep = "\t")
   # We now prepare the attribute dataframes so they can be stored in sparse format in the same data structure
   # By making sure they all have the same column names
   genres = genres.rename(columns = {"genre": "attribute", 0: "value"})
   styles = styles.rename(columns = {"style": "attribute", 0: "value"})
   labels = labels.rename(columns = {"label": "attribute", 0: "value"})
   # Here we count how many releases a band has each year
   years = release_attrs_country.groupby(by = ["band", "year"]).size().reset_index().rename(columns = {"year": "attribute", 0: "value"})
   # This aggregates the success value summing across all releases of a band
   success = release_attrs_country.groupby(by = "band")["success"].sum().reset_index().rename(columns = {"success": "value"})
   # We want to have in sparse format also all the attributes we stored in dense format, so we again rename the columns
   band_country = band_country.rename(columns = {"canonical_id": "band", "country": "value"})
   main_genre = main_genre.rename(columns = {"genre": "value"})
   avg_year = avg_year.rename(columns = {"year": "value"})
   band_names = bands[["band", "band_name"]].rename(columns = {"band_name": "value"})
   # In the sparse format, we need to store in a column the name of the attribute
   band_country["attribute"] = "origin"
   main_genre["attribute"] = "main_style"
   avg_year["attribute"] = "avg_year"
   band_names["attribute"] = "band_name"
   success["attribute"] = "success"
   # We do not distinguish between genre and style any more, so each individual genre/style attribute starts with the same "g_" prefix
   # We use these prefix so it is easier to manage attributes coming from different data when we will build the tensors
   genres["attribute"] = genres["attribute"].map(lambda x: f"g_{x}")
   styles["attribute"] = styles["attribute"].map(lambda x: f"g_{x}")
   # Each label attribute starts with "l_"
   labels["attribute"] = labels["attribute"].map(lambda x: f"l_{x}")
   # Each year attribute starts with "y_"
   years["attribute"] = years["attribute"].map(lambda x: f"y_{x}")
   # We can finally put together the sparse node attribute data structure
   nodes = pd.concat([
      band_names,
      band_country,
      main_genre,
      avg_year,
      success,
      genres,
      styles,
      labels,
      years
   ])
   sys.stderr.write("Writing node attributes...\n")
   # Still making sure we only have nodes in the actual backbones
   nodes = nodes[nodes["band"].isin(nodeset)]
   nodes.to_csv(f"datasets/{country}/nodes.tsv", index = False, sep = "\t")

sys.stderr.write("Loading bipartite network...\n")
# We read the tripartite network, which is the main structure we can reuse for all regional and temporal networks
bipartite = pd.read_csv("release_band_artist_role.tsv", sep = "\t")
# We read the release attributes, which we can reuse for the node attributes for all regional and temporal networks
release_attrs = pd.read_csv("release_attributes.tsv", sep = "\t")
# And we can use to add the year information to the tripartite network, so it is easy to filter to make different yearly cumulative snapshots
bipartite["year"] = bipartite["release"].map(release_attrs.set_index("release")["year"].to_dict())

# We do each country one by one, and then we do the global network
for country in ["global", "argentina", "australia", "austria", "belgium", "bolivia", "brazil", "bulgaria", "canada", "chile",
                "colombia", "czechoslovakia", "denmark", "finland", "france", "germany", "greece", "hungary", "iceland",
                "ireland", "italy", "jamaica", "japan", "mexico", "netherlands", "new_zealand", "norway", "peru",
                "poland", "portugal", "romania", "spain", "sweden", "switzerland", "uk", "uruguay", "us", "ussr",
                "venezuela", "yugoslavia"]:
   sys.stderr.write(f"{country}\n")
   # First we select the canonical Discogs ids of the bands we want in the current country
   band_country = select_country(country)
   # We create a subview of the tripartite network with only the bands we just selected
   bipartite_country = bipartite[bipartite["band"].isin(set(band_country["canonical_id"]))]
   nodeset = set()
   for year in range(1898, 2026):
      sys.stderr.write(f"{year}\r")
      # This function projects and backbones the tripartite network only for the given geographical region
      # with releases up until the specified year
      # It returns the ids of the nodes that made it in the final backbone
      # We save them because they tell us which are the nodes we need to generate the attributes of
      nodeset_year = write_cumulative_year(bipartite_country, year)
      # Because nodes can appear/disappear among the years, we keep a superset of all nodes
      # So that we need to generate only one node attribute file
      nodeset |= nodeset_year
   # Once we're done with the backbones, we know which nodes need attributes and we generate them
   write_node_attributes(nodeset, band_country, bipartite_country, release_attrs)



