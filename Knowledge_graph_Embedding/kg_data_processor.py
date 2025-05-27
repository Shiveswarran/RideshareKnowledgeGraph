import pandas as pd
from sklearn.model_selection import train_test_split

class KGDataProcessor:
    def __init__(self, csv_path, seed=42):
        """
        Initialize the data processor.

        Parameters:
        - csv_path: str, path to the CSV file containing 'subject', 'relation', 'object'.
        - seed: int, random seed for reproducible splitting.
        """
        self.csv_path = csv_path
        self.seed = seed
        self.df = pd.read_csv(csv_path)
        
        # Check if the required columns exist
        required_cols = {'subject', 'relation', 'object'}
        if not required_cols.issubset(set(self.df.columns)):
            raise ValueError(f"CSV file must contain the columns: {required_cols}")
        
        self._build_vocab()
        self.triplets = self._map_triplets_to_ids()

    def _build_vocab(self):
        """
        Build vocabularies for entities and relations.
        """
        # Gather all unique subjects and objects as entities
        subjects = set(self.df['subject'].unique())
        objects = set(self.df['object'].unique())
        self.entities = list(subjects.union(objects))
        
        # Get unique relations
        self.relations = list(self.df['relation'].unique())
        
        # Create mapping dictionaries
        self.entity2id = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation2id = {relation: idx for idx, relation in enumerate(self.relations)}
        
        print(f"Total unique entities: {len(self.entities)}")
        print(f"Total unique relations: {len(self.relations)}")

    def _map_triplets_to_ids(self):
        """
        Map the CSV rows into a list of triplets using the vocab dictionaries.
        
        Returns:
            A list of tuples: (subject_id, relation_id, object_id)
        """
        triplets = []
        for _, row in self.df.iterrows():
            s = row['subject']
            r = row['relation']
            o = row['object']
            triplets.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return triplets

    def split_data(self, train_ratio=0.8, valid_ratio=0.1):
        """
        Split the triplet list into train, validation, and test sets.

        Parameters:
        - train_ratio: float, ratio of samples for training.
        - valid_ratio: float, ratio of samples for validation.

        Returns:
            train_triplets, valid_triplets, test_triplets (each a list of triplets).
        """
        if train_ratio + valid_ratio >= 1.0:
            raise ValueError("train_ratio + valid_ratio must be less than 1.0")
        
        # First split into train+validation and test
        train_val, test = train_test_split(
            self.triplets, 
            test_size=1 - (train_ratio + valid_ratio), 
            random_state=self.seed
        )
        # Then split train_val into train and validation
        valid_size = valid_ratio / (train_ratio + valid_ratio)
        train, valid = train_test_split(
            train_val, 
            test_size=valid_size, 
            random_state=self.seed
        )
        print(f"Train samples: {len(train)}, Validation samples: {len(valid)}, Test samples: {len(test)}")
        return train, valid, test