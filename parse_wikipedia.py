## Code from https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c
## Adapted to retrieve species pages
## Standard library
import gc
import json
import os
import re
import subprocess
import xml.sax
from functools import partial
from itertools import chain
from multiprocessing import Pool
## Third party
import mwparserfromhell
import pandas as pd
from tqdm import tqdm

## Loading data
speciesRecords = pd.read_csv("raw_data/gbif_raw.csv", sep="\t")

## Simplifying columns
speciesRecords = speciesRecords[speciesRecords["speciesKey"].notna()]
speciesRecords = speciesRecords[["species", "speciesKey"]].drop_duplicates().reset_index().drop(["index"], axis=1)
speciesRecords["speciesKey"] = speciesRecords["speciesKey"].apply(lambda x : int(x))

def match_species(species, df=speciesRecords, data_path="./WikiSpeciesHabitats/species/"):
    """Match parsed species article to a GBIF species instance and save information in a json file"""
    title, properties, texts, text_length = species
    binomial_name = None
    #print(f"found speciesbox in page {title}")

    ## Try to create the species binomial name using genus and species, or taxon properties of the parsed infobox
    try:
        binomial_name = properties["genus"]+" "+properties["species"].split()[0]
        #print("case genus and species")
    except:
        try:
            binomial_name = properties["taxon"]
            #print("case taxon")
        except:
            pass
    if binomial_name:
        #print(f"binomial name {binomial_name}")

        ## Match binomial name to GBIF instance
        if binomial_name in df["species"].to_list():
            #print(f"matched, number of matches : {len(df[df['species']==binomial_name]['speciesKey'].to_list())}")
            ## Multiple species key can share the same binomial name (but have different scientific names)

            for speciesKey in df[df["species"]==binomial_name]["speciesKey"].to_list():
                #print(f"speciesKey : {speciesKey}")
                ## Create json file content
                content = {"binomialName":binomial_name, "speciesKey":speciesKey, "pageText":texts, "textLength":text_length, "pageTitle":title}

                ## See if a json file for this species Key already exists
                if f"{speciesKey}.json" in os.listdir(data_path):
                    #print("existing file")
                    with open(data_path + f"{speciesKey}.json", "r") as fp:
                        existing_file = json.load(fp)

                    ## Check if new text is longer and replace older text if so
                    if len(texts) > len(existing_file["pageText"]):
                        #print("replaced file")
                        with open(data_path + f"{speciesKey}.json", "w") as fp:
                            json.dump(content, fp)
                            
                ## Save new file if no previous instance is known
                else:
                    #print("new file")
                    with open(data_path + f"{speciesKey}.json", "w") as fp:
                        json.dump(content, fp)
    
    else:
        #print("could not buid binomial name for {title}")
        #print(properties)
        pass

def process_article(title, text, timestamp, template = 'Speciesbox'):
    """Process a wikipedia article looking for template"""
    
    # Create a parsing object
    wikicode = mwparserfromhell.parse(text)
    
    # Search through templates for the template
    matches = wikicode.filter_templates(matches = template)

    
    # Filter out errant matches
    matches = [x for x in matches if x.name.strip_code().strip().lower() == template.lower()]
    
    if len(matches) >= 1:
        # template_name = matches[0].name.strip_code().strip()

        # Extract information from infobox
        properties = {param.name.strip_code().strip(): param.value.strip_code().strip() 
                      for param in matches[0].params
                      if param.value.strip_code().strip()}

        # Extract texts
        texts = wikicode.strip_code().strip()
        

        # Find approximate length of article
        text_length = len(wikicode.strip_code().strip())
        #print(wikicode.strip_code().strip())

        return (title, properties, texts, text_length)

## Wikipedia pages parser
class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Parse through XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._species_count = 0
        self._article_count = 0
        self._non_matches = []

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text', 'timestamp'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self._article_count += 1
            # Search through the page to see if the page is a specie
            species = process_article(**self._values, template = 'Speciesbox')
            # Append to the list of species
            if species:
                self._species_count += 1
                match_species(species=species)

def find_species(data_path, limit = None):
    """Find all the specie articles from a compressed wikipedia XML dump.
       `limit` is an optional argument to only return a set number of species.
        If save, species are saved to partition directory based on file name"""

    # Object for handling xml
    handler = WikiXmlHandler()

    # Parsing object
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    # Iterate through compressed file
    for i, line in enumerate(subprocess.Popen(['bzcat'], 
                             stdin = open(data_path), 
                             stdout = subprocess.PIPE).stdout):
        try:
            parser.feed(line)
        except StopIteration:
            break
            
        # Optional limit
        if limit is not None and len(os.listdir("./WikiSpeciesHabitats/species/")) >= limit:
            return None

    # Memory management
    del handler
    del parser
    gc.collect()
    return None


if __name__=="__main__":
    print(f"loaded data lenght : {len(speciesRecords)}")
    # Making partitions for the multiprocessing
    root = "./wikipedia_dump/"
    partitions = [root + file for file in os.listdir(root) if 'xml-p' in file]

    ## Multiprocessing
    # Create a pool of workers to execute processes
    """pool = Pool(processes = 8)
    # Map (service, tasks), applies function to each partition
    results = pool.map(find_species, partitions)
    pool.close()
    pool.join()"""

    ## Single process (for debugging)
    for part in tqdm(partitions):
        find_species(data_path=partitions[0])

