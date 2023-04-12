import pandas as pd


class InputsTargetsBuilder():
    def __init__(self, processed_data_path, final_data_path, level="class"):
        self.processed_data_path = processed_data_path
        self.final_data_path = final_data_path
        self.level = level

    def process_data(self):
        self.species_habitats_records = pd.read_json(self.processed_data_path + "species_habitats_records.json", orient="records")
        self.habitats_data = pd.read_json(self.processed_data_path + "habitats_data.json", orient="records").set_index("TypoCH_NUM")