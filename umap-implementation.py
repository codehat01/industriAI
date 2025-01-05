import numpy as np
from umap import UMAP
from sklearn.preprocessing import StandardScaler

class UMAPReducer:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=2):
        self.scaler = StandardScaler()
        self.umap_model = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components
        )

    def fit_transform(self, features):
        scaled_features = self.scaler.fit_transform(features)
        embedding = self.umap_model.fit_transform(scaled_features)
        return embedding

    def transform(self, features):
        scaled_features = self.scaler.transform(features)
        embedding = self.umap_model.transform(scaled_features)
        return embedding

class ESGDimensionReducer:
    def __init__(self):
        self.umap_env = UMAPReducer(n_components=2)
        self.umap_social = UMAPReducer(n_components=2)
        self.umap_gov = UMAPReducer(n_components=2)

    def reduce_dimensions(self, bert_features, t5_scores):
        env_embedding = self.umap_env.fit_transform(bert_features * t5_scores[:, 0:1])
        social_embedding = self.umap_social.fit_transform(bert_features * t5_scores[:, 1:2])
        gov_embedding = self.umap_gov.fit_transform(bert_features * t5_scores[:, 2:3])
        
        return {
            'environmental': env_embedding,
            'social': social_embedding,
            'governance': gov_embedding
        }
