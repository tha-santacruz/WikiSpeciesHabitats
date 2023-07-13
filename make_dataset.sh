#! /bin /bash

python dataset_building/parse_wikipedia.py
python dataset_building/create_dataset.py --STEP all
python dataset_building/progressive_species_splitting.py