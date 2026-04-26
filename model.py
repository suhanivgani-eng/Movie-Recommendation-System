import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ensure model folder exists
os.makedirs("model", exist_ok=True)

# load dataset
movies = pd.read_csv("dataset/indian_movies.csv")

print("Dataset loaded")
print(movies.columns)

# combine features
movies["tags"] = movies["Genre"].astype(str) + " " + movies["Language"].astype(str)

print("Tags created")

# vectorization
cv = CountVectorizer(stop_words="english")

vectors = cv.fit_transform(movies["tags"]).toarray()

print("Vectorization complete")

# similarity
similarity = cosine_similarity(vectors)

print("Similarity calculated")

# save model files
pickle.dump(movies, open("model/movies.pkl","wb"))
pickle.dump(similarity, open("model/similarity.pkl","wb"))

print("Model created successfully")