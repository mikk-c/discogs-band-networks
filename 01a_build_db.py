import sys
sys.stderr.write("Setting up...\n")
import gzip, re, pickle, random
import pandas as pd
import networkx as nx
from lxml import etree

# This function essentially populates the bipartite band-artist network
# The network connects each band to the artists that worked in a specific release
def role_module():
   # The first step is to create a link between the band as a band and the band as an artist
   # This is because occasionally bands are credited as a whole, and doing so allows us to have a more precise projection
   release_roles.extend([(release_id, band_b, band_a, "Band") for band_a in bands for band_b in bands])
   # The main information about who played in the release is included in the extra artist tag
   extra_artists = elem.find("extraartists")
   if extra_artists is not None:
      # So we feed it to our specialized function that can weed out musical roles from non-musical roles
      # And that will return a list of link we append to our data structure
      release_roles.extend(add_extra_artists_credit(extra_artists))
   # Occasionally, Discogs will attach the extra artist information not to the release, but to a specific track
   # So we find the track list...
   tracks = elem.find("tracklist")
   if tracks is not None:
      tracks = tracks.findall("track")
      # ...and we loop over the tracks, if we found any
      for track in tracks:
         extra_artists = track.find("extraartists")
         if extra_artists is not None:
            # Each track extra artist data structure is identical to what was attached to the release
            # So we can re-use the same function
            release_roles.extend(add_extra_artists_credit(extra_artists))

def year_module():
   # We keep a list of all the possible years of this release
   possible_years = []
   # First we look if the tag is populated
   released_tag = elem.find("released")
   if released_tag is not None:
      released_tag = released_tag.text.strip()[:4]
      if released_tag != "":
         try:
            # If it is populated, valid, and not empty, we trust it and return it
            return int(released_tag)
         except ValueError:
            pass
   # We're here if we didn't get a proper year from the XML tag and we need to check the notes
   # We now run our regular expression to find copyright information from the release's description...
   possible_years.extend([extract_year_from_text(description.text) for description in elem.iter("description")])
   # ...and notes
   notes_tag = elem.find("notes")
   if notes_tag is not None:
      possible_years.append(extract_year_from_text(notes_tag.text))
   # We filer out false positives, years of release that could not possibly be true
   possible_years = [y for y in possible_years if y is not None and y != "" and y > 1888 and y <= 2025]
   # If we have at least one possible year of release, we pick the smalles one taking the minimum (since these are stored as int)
   if len(possible_years) > 0:
   	return min(possible_years)

# This is a list we need to have a single separate word matching exactly
# Because they are short strings that could be contained in an unwanted role
exact_wanted_roles = ["rap", "oud", "ney", "coro", "koto", "saw", "viol", "arpa", "bells", "udu",
                      "sho", "pipa", "bones", "mc", "tres", "sampler", "leader", "dhol", "viol",
                      "bombo", "spoons", "cura", "jug"]

# This is a list of instruments and musical roles we allow to create an edge in the network
# These are to be sondiered fuzzy matches: they resolve positively if they are contained in a string even if it is in a subword
wanted_roles = ["voic","voca","musician","performer","scratch","vocoder","e-bow","programmed",
          "guitar","sitar","banjo","dobro","mandolin","bass","drum","tom tom","table","organ",
          "keyboard","pian","rhodes","mellotron","synth","violin","viola","percussion","handclaps",
          "hihat","pandeiro","cajon","tamb","bata","prod","mixing","mixed","saxophone","accordion",
          "trumpet","flute","trombone","horn","whistle","recorder","harmonica","oboe","cello",
          "strings","conductor","arranged","orchestra","choir","ukulele","harp","harmonium","marimba",
          "vibraphone","cymbal","clarinet","whistl","tuba", "congas", "bongos", "featuring", "feat",
          "reeds","psaltery","concertina","brass","timbales","beatbox","gong","clarinet","bouzouki",
          "conductor","gittern","clavinet","marxophone","melodica","kora","mridangam","schwyzerörgeli",
          "fiddle","hurdy gurdy","bandoneon","kalimba","pipe","dulcimer","lute","yodel","timpani",
          "wind", "dizi", "erhu", "gaohu", "huqin", "ruan", "sheng", "yangqin", "bugle", "taragot",
          "violon", "zither", "fluier", "cimbal", "tárogató", "soloist", "electronic", "kemenche",
          "beat", "band", "accompanied", "directed", "santur", "santoor", "lyre", "cittern",
          "laouto", "baglama", "kanun", "tzouras", "outi", "cornet", "duduk", "sounds", "theorbo",
          "chorus", "ensemble", "guest", "quena", "tabla", "fife", "bendir", "davul", "chanter",
          "instruments", "kaval", "glockenspiel", "cor anglais", "dulcian", "sackbut", "theremin",
          "rap [uncredited]", "dj mix", "music director", "traverso", "chitarrone", "didgeridoo",
          "sousaphone", "tam-tam", "clavichord", "bombarde", "xylophone", "celesta", "shakuhachi",
          "djembe", "vielle", "baryton", "charango", "bodhrán", "alto viol", "castanets", "balafon",
          "snare", "triangle", "ocarina", "blaster beam", "surdo", "afoxé", "melodeon", "euphonium",
          "cuica", "maracas", "launeddas", "shawm", "rappers", "washboard", "rebec", "balalaika",
          "cowbell", "hardingfele", "guiro", "spinet", "guzheng", "kazoo", "mellophone", "clarion",
          "zurna", "zongora", "cobza", "ütőgardon", "guitalele", "musette", "chimes", "gadulka",
          "finger snaps", "gamelan", "omnichord", "halldorophone", "bansuri", "gaita", "mandola",
          "vihuela", "sarangi", "langeleik", "saxello", "mbira", "flugabone", "guimbri", "rap, rap",
          "darbouka", "fujara", "chalumeau", "rhythm section", "shaker", "noises", "rap [add",
          "lyricon", "talkbox", "bajo sexto", "cabasa", "cuatro", "drone", "tenor viol", "treble viol",
          "tubular bells", "oud [aoud]", "sarod", "cavaquinho", "agogô", "berimbau", "ganzá", "coro, coro",
          "tan-tan", "repinique", "reco-reco", "zabumba", "rabeca", "crotales", "caixa", "shamisen",
          "taiko", "biwa", "kokyu", "taishōgoto", "shinobue",  "hayashi", "sanshin", "toasting",
          "leader [string", "cajón", "rap [mcs]", "lirone", "oud, oud", "helicon", "kantele", "claves",
          "dhol [νταούλι]", "laúd", "cümbüş", "toubeleki", "leader [leitung]", "rap [mit]", "viol [alto]",
          "viol [gambe]", "carillon", "domra", "skratjie", "viol [viole]", "viol [solo]", "chapman stick",
          "viol [violen]", "leader, leader", "viol, viol", "bells [hand", "dholak", "shenai", "humming",
          "sintir", "leader [řídí]", "leader [primáš]", "gusli", "rap [gościnnie]", "kamancha", "dilruba",
          "guqin", "bayan", "singing bowls", "kanklės", "sopilka", "clap sticks", "esraj",
          "caja vallenata", "guacharaca", "coro [coros]", "tiple", "bajo quinto", "acordeonero", "dulzaina",
          "timple", "chirimia", "txistu", "jarana", "marimbula", "musical bow", "flageolet", "cantor",
          "tipple", "bells [inca]", "metallophone", "sampona", "jouhikko", "leader [kor", "leader [orkester",
          "conch [guarura]", "sho, sho", "pipa [pipá]", "chamberlin", "coro [coro femenino]", "caxixi",
          "arpa, arpa", "ronroco", "moon bells"]

# This is a list of unwanted roles, these roles cannot create and edge in the network
# This list is not used by the script, because there is no difference between an unwanted role and a role that we haven't classified as wanted
# However having an explicit blacklist is helpful to weed out roles that should never be included from those that could, in case someone wants to re-run the code with different choices
unwanted_roles = ["cover", "engineer","photography","design","art","lacquer","written","lyrics",
                  "managment","master","recorded", "a&r","liner notes","sleeve notes","layout", "technician",
                  "illustration","management","booking","music by","edited by","songwrit","make-up", "model",
                  "merchand", "legal", "remix", "words", "composed", "author", "text", "assistance", "read by",
                  "supervised", "narrator", "painting", "compiled", "logo", "field recording", "film director",
                  "studio ambience", "libretto", "coordinator", "translated", "transferred", "adapted", "graphics",
                  "recording supervisor", "sleeve", "styling", "typography", "make up", "copyist", "stylist",
                  "stage manager", "presenter", "hair", "interviewee", "speech", "booklet editor", "film editor",
                  "editor [", "reviewer", "interviewer", "cameraman", "lighting", "camera operator", "plated by",
                  "video editor", "music consultant", "script by", "lithography", "project manager", "publisher",
                  "research", "instrument builder", "promotion", "tour manager", "bus drivers", "advisor", "vj",
                  "repetiteur", "crew", "cadenza", "score editor", "transcription", "contractor", "commentator",
                  "dialog", "music librarian", "choreography", "filmed by", "restoration", "drawing", "sequenced by",
                  "concept by", "luthier", "public relations", "animation", "video director", "screen printing",
                  "overdubbed", "consultant", "lettering", "screenwriter", "administrator", "image editor",
                  "commission", "proofreader", "calligraphy", "cinematographer", "footage", "gaffer", "collected",
                  "tap dance", "dancers", "actor", "lathe cut", "creative director", "hosted by", "marketing",
                  "tape op", "tracking", "thanks", "equipment", "realization", "costume", "catering", "clothes",
                  "videography", "dress", "choregrapher", "recenzent", "proofreading", "sponsor", "scenographer",
                  "gastronomy", "tour promoter", "photograph", "special fx", "special effects"]

# This function gets a tag containing all the credits for a release and weeds out the non musical role
# It returns a list of (release, band, artist, role) tuple only for the valid musical roles
def add_extra_artists_credit(extra_artists):
   credits = []
   # We find all the artists credited
   extra_artists = extra_artists.findall("artist")
   for extra_artist in extra_artists:
      # We find the id, we use the one listed, and we will solve the issue of aliases later in the script
      extra_artist_id = extra_artist.find("id")
      if extra_artist_id is not None:
         extra_artist_id = int(extra_artist.find("id").text)
         # Now we extract the information of what this artist did in the release
         extra_artist_role = extra_artist.find("role")
         if extra_artist_role is not None:
            # We make sure everything is in lowercase to make the text check easier
            extra_artist_role = extra_artist_role.text.replace('\t', '').lower()
            # We loop over all roles of the artist and we include the connection only if either of two conditions is met:
            # Either the role description contains any wanted role from the wanted_roles list
            # Or there is an exact match with one element from the exact_wanted_roles list
            # This ensures we are actually looking at a musical role
            if any(elem in extra_artist_role for elem in wanted_roles) or any(extra_artist_role == elem for elem in exact_wanted_roles):
               credits.extend([(release_id, band_b, extra_artist_id, extra_artist_role) for band_b in bands])
   return credits

# Base year pattern
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Each tuple = (keyword pattern, number of characters forward to scan)
KW_PATTERNS = [
    (re.compile(r"\breleased\b", re.I), 80),
    (re.compile(r"\brelease\s+date\b", re.I), 80),
    (re.compile(r"\brecorded\b", re.I), 80),
    (re.compile(r"\bissued\b", re.I), 80),
    (re.compile(r"\bcopyright\b", re.I), 160),
    (re.compile(r"(?:©|\(c\)|\([cC]\))", re.I), 200),  # (C) / © / (c)
]

# The function takes some text which might contain year information an returns it
def extract_year_from_text(text):
    if not text:
        return None
    # Making sure we find the copyright symbold as unicode
    s = text.replace("\u00A9", "©")
    # We scan all possible patterns we have in our regexp engine
    for pattern, window in KW_PATTERNS:
        for m in pattern.finditer(s):
            start = m.end()
            segment = s[start:start + window]
            y = YEAR_RE.search(segment)
            # If we find a match we return it as an integer
            if y:
                return int(y.group(0))
    return None

sys.stderr.write("Load success info...\n")
release_success = pd.read_csv("releases_have_want.tsv", sep = "\t")
release_success["success"] = release_success["have"] + release_success["want"]
release_success = release_success.set_index("release")["success"].to_dict()

# We want to create a network of aliases, two aliases connected if they belong to the same entity
# Then each connected component in the network is a unique entity
sys.stderr.write("Building entity alias graph...\n")
alias_edges = set()
names = {}

# We start by iterating over the artists file, each entry here is a separate artist
# Bands are artists too in this file
i = 0
with gzip.open("raw_data/discogs_20251101_artists.xml.gz", "rb") as f:
   context = etree.iterparse(f, events = ("end",), tag = "artist")
   # We iterate over the artist tags
   for event, elem in context:
      # We find the id of the current artist
      band_discogs_id = int(elem.find("id").text.strip())
      # Here we store the name of the artist as displayed on Discogs
      name = elem.find("name")
      # If Discogs has a recorded name, we strip out whitespaces
      if name is not None:
         names[band_discogs_id] = name.text.strip()
      # We find all aliases of this artist known by Discogs
      band_aliases = elem.find("aliases")
      if band_aliases is not None:
         for alias in band_aliases.findall("name"):
            alias_id = int(alias.attrib["id"])
            # We add pairs to the alias edge set we're building, connecting the artist id with all its aliases
            alias_edges.add((alias_id, band_discogs_id))
      elem.clear()
      while elem.getprevious() is not None:
         del elem.getparent()[0]
      i += 1
      if i % 32768 == 0:
         sys.stderr.write(f"{100*i/9798352:.2f}%\r")

sys.stderr.write("Building label alias graph...\n")
label_alias_edges = set([])

# We now iterate over the labels file, each entry here is a separate label
i = 0
with gzip.open("raw_data/discogs_20251101_labels.xml.gz", "rb") as f:
   context = etree.iterparse(f, events = ("end",), tag = "label")
   # We iterate over the label tags
   for event, elem in context:
      # We find the id of the current label
      discogs_id = elem.find("id")
      if discogs_id is not None:
         discogs_id = int(discogs_id.text.strip())
         # We collect the list of all sublabels known to Discogs
         sublabels = elem.find("sublabels")
         if sublabels is not None:
            sublables = sublabels.findall("label")
            for sublabel in sublabels:
               if "id" in sublabel.attrib:
                  # If we have an ID for the sublabel, we create a label-sublabel ID pair connecting them in a graph
                  label_alias_edges.add((discogs_id, int(sublabel.attrib["id"])))
         # We also look for a parent label
         parent = elem.find("parentLabel")
         if parent is not None:
            # And we add the edge connecting the label to its parent
            label_alias_edges.add((int(parent.attrib["id"]), discogs_id))
      elem.clear()
      while elem.getprevious() is not None:
         del elem.getparent()[0]
      i += 1
      if i % 32768 == 0:
         sys.stderr.write(f"{100*i/2552038:.2f}%\r")

release_year = []
release_roles = []
release_labels = []
release_genres = []
release_styles = []

sys.stderr.write("Parsing releases...\n")
i = 0
with gzip.open("raw_data/discogs_20251101_releases.xml.gz", "rb") as fin:
   context = etree.iterparse(fin, events = ("end",), tag = "release")
   for event, elem in context:
      bands = elem.find("artists")
      if bands is not None:
         bands = list(map(int, [x.text for x in bands.iter("id")]))
         if len(bands) > 0:
            release_id = int(elem.attrib["id"])
            # This function finds the year of the release
            year = year_module()
            # If we could not find a valid year, we skip the release, as we cannot place it in our timeline
            if year is not None:
               # This function populates the release_roles list with tripartite release-band-artist edges
               # If the artist worked with the band on a given release
               role_module()
               release_labels.extend([(release_id, int(label.attrib["id"])) for label in elem.iter("label") if "id" in label.attrib])
               release_genres.extend([(release_id, genre.text.strip().lower().replace(',', '_').replace(' ', '_')) for genre in elem.iter("genre")])
               release_styles.extend([(release_id, style.text.strip().lower().replace(',', '_').replace(' ', '_')) for style in elem.iter("style")])
               # Recording the year of the release as we found it using our year module
               release_year.append((release_id, year))
      elem.clear()
      while elem.getprevious() is not None:
         del elem.getparent()[0]
      i += 1
      if i % 32768 == 0:
         sys.stderr.write(f"{100*i/18647889:.2f}%\r")

sys.stderr.write("Creating data structures...\n")
# This will be the final tripartite network connecting bands and artists via releases
release_roles = pd.DataFrame(data = release_roles, columns = ("release", "band", "artist", "role"))
# Each of the following dataframes attaches different information to each release
# Which will be at the basis for the node attributes
# Using the tripartite network to aggregate the data from all of each band's releases
release_year = pd.DataFrame(data = release_year, columns = ("release", "year"))
release_labels = pd.DataFrame(data = release_labels, columns = ("release", "label"))
release_genres = pd.DataFrame(data = release_genres, columns = ("release", "genre"))
release_styles = pd.DataFrame(data = release_styles, columns = ("release", "style"))

# We add the success information to the release attributes
release_year["success"] = release_year["release"].map(release_success).fillna(0).astype(int)
# We need to add the success information to the tripartite network and the labels, to aid for finding the canonincal ids
release_roles["success"] = release_roles["release"].map(release_success).fillna(0).astype(int)
release_labels["success"] = release_labels["release"].map(release_success).fillna(0).astype(int)

# We need an "alias -> success" data structure, connecting each artist ID with the success count
# We use the artist column because it is a superset including the bands as well (by construction)
alias_success = release_roles.groupby(by = "artist")["success"].sum().to_dict()
# We need a "label -> success" data structure here, connecting each label ID with the success count
label_success = release_labels.groupby(by = "label")["success"].sum()
label_success.reset_index().to_csv("label_success.tsv", index = False, sep = "\t")
label_success = label_success.to_dict()

sys.stderr.write("Disambiguating entities...\n")
# We need to decide what is the canonical id and name of an entity among all aliases.
# We will pick the alias with most records own+want on Discogs     
aliases = {}
# First we need to transform the list of tuples in a networkx object
alias_edges = nx.from_edgelist(list(alias_edges))

# We iterate over all connected components of the graph
# Each connected component is a single artist, and the nodes in the components are all of their aliases
for ccomp in nx.connected_components(alias_edges):
   # We create a subdictionary with the success of all aliases in the component
   alias_success_inccomp = {node: alias_success[node] for node in ccomp if node in alias_success}
   # If we found at least one alias with success information...
   if len(alias_success_inccomp) > 0:
      # ...we use the one with the highest success as the canonical ID
      canonical_id = max(alias_success_inccomp, key = alias_success_inccomp.get)
   else:
      # ...otherwise we'll pick a random one
      canonical_id = random.choice(tuple(ccomp))
   # Now the aliases dictionary will have for key all ids in the connected component...
   # ...and for value the canonical id we just picked
   aliases.update({node: canonical_id for node in ccomp})

# We save the dictionary so we can re-use it, if needed
with open(f"aliases_entities", 'wb') as f:
   pickle.dump(aliases, f, protocol = pickle.HIGHEST_PROTOCOL)

sys.stderr.write("Disambiguating labels...\n")
# We need to decide what is the canonical id and name of a label among all aliases.
# We will pick the alias with most records own+want on Discogs
label_aliases = {}
label_alias_edges = nx.from_edgelist(list(label_alias_edges))
# Remove nodes that are actually not a label
label_alias_edges.remove_nodes_from([1818, 212])
# Add edges to fix loops
label_alias_edges.add_edges_from([(38404, 19855),])
# Remove edges to fix loops
label_alias_edges.remove_edges_from([(572946, 225987), (448050, 862360), (267267, 37258), (109358, 5195), (45337, 42667), (587087, 19154)]) 

# We iterate over all connected components of the graph
# Each connected component is a single ownership structure with the same root, and the nodes in the components are all of their sublabels
for ccomp in sorted(nx.connected_components(label_alias_edges), key = len, reverse = True):
   # We create a subdictionary with the success of all aliases in the component
   alias_success_inccomp = {node: label_success[node] for node in ccomp if node in label_success}
   # If we found at least one alias with success information...
   if len(alias_success_inccomp) > 0:
      # ...we use the one with the highest success as the canonical ID
      canonical_id = max(alias_success_inccomp, key = alias_success_inccomp.get)
   else:
      # ...otherwise we'll pick a random one
      canonical_id = random.choice(tuple(ccomp))
   # Now the aliases dictionary will have for key all ids in the connected component...
   # ...and for value the canonical id we just picked
   label_aliases.update({node: canonical_id for node in ccomp})

# We save the dictionary so we can re-use it, if needed
with open(f"aliases_labels", 'wb') as f:
   pickle.dump(label_aliases, f, protocol = pickle.HIGHEST_PROTOCOL)

sys.stderr.write("Writing to disk...\n")
# Now that we know the canonical ids of all entities, we need to replace all ids with their canonical id
release_roles["band"] = release_roles["band"].map(lambda x: aliases[x] if x in aliases else x)
release_roles["artist"] = release_roles["artist"].map(lambda x: aliases[x] if x in aliases else x)
release_labels["label"] = release_labels["label"].map(lambda x: label_aliases[x] if x in label_aliases else x)

# We create an aggregated attribute file that for each band reports its aggregate success
band_attributes = release_roles[["release", "band", "success"]].drop_duplicates()
band_attributes = band_attributes.groupby(by = "band")["success"].sum().reset_index()
band_attributes["name"] = band_attributes["band"].map(names)
band_attributes.to_csv("band_attributes.tsv", sep = "\t", index = False)

# We do not store information about the specific roles, to avoid double counting an artist for a given release
release_roles = release_roles[["release", "band", "artist"]].drop_duplicates()

release_year.to_csv("release_attributes.tsv", sep = "\t", index = False)
release_roles.to_csv("release_band_artist_role.tsv", sep = "\t", index = False)
release_labels.drop("success", axis = 1).to_csv("release_labels.tsv", sep = "\t", index = False)
release_genres[release_genres["genre"] != "1"].to_csv("release_genres.tsv", sep = "\t", index = False)
release_styles.to_csv("release_styles.tsv", sep = "\t", index = False)


