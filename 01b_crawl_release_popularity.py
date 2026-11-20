import discogs_client, time, requests
import pandas as pd

# This portion needs to be run once manually to authorize the client
#client.get_authorize_url()
#code = ""
#client.get_access_token('')

# Here you need to include the authentication information for the client used
client = discogs_client.Client(
   "",
   consumer_key = "",
   consumer_secret = "",
   token = u"",
   secret = u""
)

# In case of crashes, we read back the information we already crawled,
# to skip releases we don't need to query again
already_done = set(pd.read_csv("releases_have_want.tsv", sep = '\t')["release"])

# We loop over a simple file (ids_todo) which has a release id per line
# And then we append to the tsv file with the releases we have collected so far
with open("ids_todo", 'r') as fin, open(f"releases_have_want.tsv", 'a') as fout:
   for line in fin:
      # We get the id of the current release we want to crawl
      release_id = int(line.strip())
      # WE only actually make the API call if we haven't seen this id before
      if not release_id in already_done:
         try:
            # We're in this branch if we got a proper response from the server
            release = client.release(release_id).community
            fout.write(f"{release_id}\t{release.have}\t{release.want}\n")
            fout.flush()
         except ValueError:
            # This might happen if there was a problem in crawling
            # We store a -1 to be then manually inspected later to figure out what happened
            # In most cases this was a glitch and we get the info when we re-crawl,
            # removing the -1 ids so they're crawled again
            # In some cases, this is because the release was deleted between the XML dump and the API call
            fout.write(f"{release_id}\t-1\t-1\n")
            fout.flush()
         except discogs_client.exceptions.HTTPError:
            # Same as the previous block
            fout.write(f"{release_id}\t-1\t-1\n")
            fout.flush()
         except requests.exceptions.ConnectionError:
            # We're in this block if we were hit by a rate limit
            # This should not happen given that we're respecting the rate limits
            # But occasionally it does, so to be safe, we wait a bit longer
            # Note how nothing is written and the id is skipped, so this will be picked up by a following run
            time.sleep(10)
            continue
         time.sleep(1)
