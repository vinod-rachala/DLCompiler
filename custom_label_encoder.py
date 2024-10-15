# custom_label_encoder.py

import numpy as np
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(LabelEncoder):
    def __init__(self, unknown_value=-1):
        super().__init__()
        self.unknown_value = unknown_value

    def fit(self, y):
        """Fit label encoder"""
        super().fit(y)
        # Store known classes
        self.classes_ = np.append(self.classes_, '<UNK>')  # Add a token for unknowns
        return self

    def transform(self, y):
        """Transform labels, handling unknown values"""
        # Check for unknown labels and replace them with a custom unknown_value
        transformed = []
        for label in y:
            if label in self.classes_:
                transformed.append(super().transform([label])[0])
            else:
                transformed.append(self.unknown_value)
        return np.array(transformed)

    def inverse_transform(self, y):
        """Inverse transform labels"""
        # Inverse transform with handling of unknown values
        inv_transformed = []
        for value in y:
            if value == self.unknown_value:
                inv_transformed.append('<UNK>')
            else:
                inv_transformed.append(super().inverse_transform([value])[0])
        return np.array(inv_transformed)
