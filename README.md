# WikiSpeciesHabitats dataset
This repository containts the scripts to create the WikiSpeciesHabitats dataset. 
This dataset contains habitats and species occurence pairs, with textual descriptions of the species which are their wikipedia page text. 
Its range is limited to Switzerland because of the coverage of habitat maps.


It is obtained by merging information from three initial sources :
1. [The Habitat Map of Switzerland v1](https://www.envidat.ch/dataset/habitat-map-of-switzerland)
2. [Global Biodiversity Information Facility occurence data](https://www.gbif.org/en/occurrence/search?occurrence_status=present&q=)
3. [Wikipedia dump](https://en.wikipedia.org/wiki/Wikipedia:Database_download)


---

## Dataset building

### 1 Packages

First, install the required packages in your environments by running

```Bash
pip install -r requirements.txt
```
### 2 Study Area

1. Define a study area, and save it in the shapefile format with the name studyArea.shp. 
2. You can also create a .geojson version of the file as it can be used to extract data.
3. Save your files in the ./raw_data/studyArea/ directory

### 4 Habitat maps
1. Download the cantonal habitat maps for each canton you are interested in
2. Each download should be named habitatmap_xx_yyyymmdd, with xx being the two letter code for the canton.
3. Rename each downloaded folders as habitatmap_xx
4. Place your downloaded folders in ./raw_data/.

### 5 Species occurences
1. Select the filters you want (country, administrative area, etc...), and download the occurence data in .csv format. You should choose the "simple" option.
2. rename your download as gbif_raw.csv and place it in the ./raw_data/ directory

### 6 Wikipedia articles
1. Download a Wikipedia dump ([this post](https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c) might help you)
2. Put all the dump files in the ./wikipedia_dump/ directory
3. Make sure that the ./WikiSpeciesHabitats/species/ directory exits. If not, create it.
4. Run the wikipedia dump parsing script 
```Bash
python parse_wikipedia.py
```

At this stage, your raw_data/ directory should look like this :

  ```bash
  raw_data                                      # Main Folder
      ├── habitatmap_vd                         # Controllers
      ├── flocking_crossing_controller	
  
  ```


###
