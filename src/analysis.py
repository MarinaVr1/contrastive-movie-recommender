import numpy as np
from numpy import linalg as la
from collections import Counter

def normalize(x):
    norm = la.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norm
def cosine_similarity(u, i):
    return np.sum(u * i, axis=1)

def load_movie_titles(path='u.item'):
    movies = {}
    
    with open(path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            movie_id = int(parts[0]) - 1
            title = parts[1]
            movies[movie_id] = title
            
    return movies


def load_user_info(path='u.user'):
    users = {}
    
    with open(path) as f:
        for line in f:
            user_id, age, gender, occupation, zip_code = line.strip().split('|')
            users[int(user_id)-1] = {
                'age': age,
                'gender': gender,
                'occupation': occupation
            }
            
    return users


def load_movies_with_genres(path='u.item'):
    movies = {}
    
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    with open(path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            movie_id = int(parts[0]) - 1
            title = parts[1]
            
            genre_flags = list(map(int, parts[5:]))
            genres = [
                genre_names[i]
                for i in range(len(genre_flags))
                if genre_flags[i] == 1
            ]
            
            movies[movie_id] = {
                "title": title,
                "genres": genres
            }
    
    return movies

def show_user_profile(user_id, data, movies_with_genres):
    train_items = data.train_df[
        data.train_df['user_id'] == user_id
    ]['item_id'].values
    
    genre_counter = Counter()
    
    for item in train_items:
        genres = movies_with_genres[item]["genres"]
        genre_counter.update(genres)
    
    print("\nDominantni zanrovi korisnika:")
    for genre, count in genre_counter.most_common(5):
        print(f"{genre}: {count}")


def show_recommendation_genres(top_k, movies_with_genres):
    print("\nZanrovi preporucenih filmova:")
    genre_counter = Counter()
    
    for idx in top_k:
        genres = movies_with_genres[idx]["genres"]
        genre_counter.update(genres)
    
    for genre, count in genre_counter.most_common():
        print(f"{genre}: {count}")

def recommend_for_user(user_id,user_emb,item_emb,data,movie_titles,user_info,k=10):
    user_norm = normalize(user_emb)
    item_norm = normalize(item_emb)
    u = user_norm[user_id]
    scores = item_norm @ u
    train_history = data.train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    if user_id in train_history:
        scores[list(train_history[user_id])] = -np.inf
    top_k = np.argsort(scores)[-k:][::-1]
    user_data = user_info[user_id]
    print("\n==============================")
    print(f"User {user_id}")
    print(f"Age: {user_data['age']}")
    print(f"Gender: {user_data['gender']}")
    print(f"Occupation: {user_data['occupation']}")
    
    test_item = data.test_df[
        data.test_df['user_id'] == user_id
    ]['item_id'].values[0]
    print("\nTest film:")
    print("->", movie_titles[test_item])
    print("\nTop 10 preporuka:")
    seen_titles = set()
    for idx in top_k:
        title = movie_titles[idx]
        if title not in seen_titles:
            print("->", title)
            seen_titles.add(title)
    print("==============================\n")
    
    return top_k