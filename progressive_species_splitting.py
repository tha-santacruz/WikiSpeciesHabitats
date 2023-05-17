import pandas as pd


if __name__ == "__main__":
    ## Read half2 data
    half_2_data = pd.read_json("./final_data/L1_species_based_test_data.json", orient="records")
    ## Find a splitting that ensures a high overlapping of species between both equally long splits
    max_overlap = 0
    best_rs = 0
    for i in range(100):
        t1 = half_2_data.sample(frac=0.5, random_state=i)
        t2 = half_2_data.drop(t1.index)
        t1_spe = t1["species_key"].to_list()
        t1_spe = [item for sublist in t1_spe for item in sublist]
        t1_spe = list(set(t1_spe))
        t2_spe = t2["species_key"].to_list()
        t2_spe = [item for sublist in t2_spe for item in sublist]
        t2_spe = list(set(t2_spe))
        overlap = [item for item in t1_spe if item in t2_spe]
        if len(overlap) > max_overlap:
            max_overlap = len(overlap)
            best_rs = i
    ## Get best splits and species list
    ## This one is gonna be test split
    t1 = half_2_data.sample(frac=0.5, random_state=best_rs)
    ## This one is gonna be used as a pool for train examples
    t2 = half_2_data.drop(t1.index)
    t1_spe = t1["species_key"].to_list()
    t1_spe = [item for sublist in t1_spe for item in sublist]
    t1_spe = list(set(t1_spe))
    t2_spe = t2["species_key"].to_list()
    t2_spe = [item for sublist in t2_spe for item in sublist]
    t2_spe = list(set(t2_spe))
    ## This one is gonna be used as a pool for train examples
    t3 = pd.read_json("./final_data/L1_species_based_train_data.json", orient="records").sample(len(t1), random_state=42)
    ## To log splits stats
    splits_stats = pd.DataFrame()
    ## To store chosen examples from t3
    t3_subset = t3
    ## To store unused examples from t2
    t2_pool = t2
    ## To store chosen examples from t2
    t2_subset = pd.DataFrame(columns=t2.columns)
    ## Number of division steps
    num_steps = 10
    random_state = 42
    for i in range(0,num_steps+1):
        frac = i/num_steps
        if frac > 0:
            ## Remove examples from t3 subset
            t3_subset = t3_subset.drop(t3_subset.sample(frac = 1/(num_steps+1-i), random_state = random_state).index)
            ## Add examples to t2 subset and remove them from pooling
            t2_sampled = t2_pool.sample(frac = 1/(num_steps+1-i), random_state = random_state)
            t2_pool = t2_pool.drop(t2_sampled.index)
            t2_subset = pd.concat([t2_subset, t2_sampled])
        ## Create train/val split
        t4 = pd.concat([t3_subset, t2_subset])
        ## Compute number of species in intersection, union, etc...
        t4_spe = t4["species_key"].to_list()
        t4_spe = [item for sublist in t4_spe for item in sublist]
        t4_spe = list(set(t4_spe))
        intersection = [spe for spe in t4_spe if spe in t1_spe]
        union = list(set(t4_spe + t1_spe))
        overlap = len(intersection)/len(union)
        test_fraction = len(intersection)/len(t1_spe)
        ## Log metrics
        splits_stats = pd.concat([splits_stats, pd.DataFrame(data=[frac,1-frac,len(intersection),len(union),len(t4_spe),len(t1_spe)]).T])
        ## Put 5% of examples in val
        t5 = t4.sample(frac=0.05, random_state=random_state)
        t4 = t4.drop(t5.index)
        ## Save train and val
        t4.reset_index(drop=True).to_json(f"./final_data/L1_progressive_{int(frac*100)}%_train_data.json", orient="records")
        t5.reset_index(drop=True).to_json(f"./final_data/L1_progressive_{int(frac*100)}%_val_data.json", orient="records")
    ## Save metrics
    splits_stats.columns=["half1_frac","half2_frac","species_inter","species_union","train_species","test_species"]
    splits_stats.to_json("progressive_splits_stats", orient="records")
    print(splits_stats)
    ## Save test
    t1.reset_index(drop=True).to_json("./final_data/L1_progressive_test_data.json")

    