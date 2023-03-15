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
2. It is recommended to also create a .geojson version of the file as it can be used to extract data.
3. Save your files in the raw_data/studyArea/ directory

### 3 Habitat maps
1. Download the cantonal habitat maps for each canton you are interested in
2. Each download should be named habitatmap_xx_yyyymmdd, with xx being the two letter code for the canton.
3. Rename each downloaded folders as habitatmap_xx
4. Place your downloaded folders in raw_data/.

At this stage, your raw_data/ directory should look like this :

  ```bash
  ├── raw_data                       # Main code
  │   ├── controllers                         # Controllers
  │   │   ├── flocking_crossing_controller	
  │   │   ├── flocking_obstacle_controller
  │   │   ├── formation_graph_crossing_controller
  │   │   ├── formation_graph_obstacle_controller
  │   │   └── supervisor						
  │   └── worlds                              # Default testing world
  ├── localization_library                    # Shared localization utilities
  ├── Matlab                                  # Visualization utilities
  │   ├── localization                        # Localization result visualization
  │   └── metric_computation                  # Flocking result visualization
  └── supplemental                            # Supplemental code, *not* necessary for metrics evaluation
      ├── controllers
      │   ├── crossing_pso_controller         # PSO
      │   ├── obstacle_pso_controller         # PSO
      │   ├── pso_crossing_supervisor         # PSO
      │   ├── pso_obstacle_supervisor         # PSO
      │   ├── localization_controller         # Localization
      │   └── localization_supervisor         # Localization
      └── worlds                              # Ad-hoc worlds
  ```


###
