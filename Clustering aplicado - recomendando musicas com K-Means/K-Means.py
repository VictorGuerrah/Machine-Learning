from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, euclidean_distances
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spotipy

def recommend_id(playlist_id):
  url = []
  name = []
  for i in playlist_id:
        track = sp.track(i)
        url.append(track["album"]["images"][1]["url"])
        name.append(track["name"])
  return name, url

def visualize_songs(name, url):

    plt.figure(figsize=(15,10))
    columns = 5

    for i, u in enumerate(url): 
        ax = plt.subplot(len(url) // columns + 1, columns, i + 1)
        image = io.imread(u)
        plt.imshow(image)
        ax.get_yaxis().set_visible(False)
        plt.xticks(color = 'w', fontsize = 0.1)
        plt.yticks(color = 'w', fontsize = 0.1)
        plt.xlabel(name[i], fontsize = 8)
        plt.tight_layout(h_pad=0.7, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.tick_params(bottom = False)
        plt.grid(visible=None)
    plt.show()

SEED = 1224
np.random.seed(SEED)

raw_music_data = pd.read_csv('C:/Users/victo/OneDrive/Área de Trabalho/Victor/Bathtub/Repository/Clustering aplicado - recomendando musicas com K-Means/Dados_totais.csv')
raw_genre_data = pd.read_csv('C:/Users/victo/OneDrive/Área de Trabalho/Victor/Bathtub/Repository/Clustering aplicado - recomendando musicas com K-Means/data_by_genres.csv')
raw_year_data = pd.read_csv('C:/Users/victo/OneDrive/Área de Trabalho/Victor/Bathtub/Repository/Clustering aplicado - recomendando musicas com K-Means/data_by_year.csv')

treated_music_data = raw_music_data.copy()
treated_music_data.drop(columns=["id", "name", "explicit", "key", "mode", "artists_song"], inplace=True, axis=1)
treated_music_data_without_artists = treated_music_data.drop('artists', axis=1)

treated_genre_data = raw_genre_data.copy()
treated_genre_data.drop(columns=["key", "mode", "genres"], inplace=True, axis=1)

treated_year_data = raw_year_data.copy()
treated_year_data.drop(columns=["key", "mode"], inplace=True, axis=1)
treated_year_data = treated_year_data[treated_year_data["year"] >= 2000]

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=0.7))])

# genre_embendding_pca = pca_pipeline.fit_transform(treated_genre_data)
# genre_projection = pd.DataFrame(data=genre_embendding_pca)

# model_genre = KMeans(n_clusters=8)
# model_genre.fit(genre_projection)
# prediction_genre = model_genre.predict(genre_projection)
# treated_genre_data['cluster'] = prediction_genre
# genre_projection['cluster'] = prediction_genre
# genre_projection['genres'] = raw_genre_data['genres']

# print(pca_pipeline[1].explained_variance_ratio_.sum())
# s_score = silhouette_score(genre_embendding_pca, prediction_genre, metric='euclidean')
# db_score = davies_bouldin_score(genre_embendding_pca, prediction_genre)
# ch_score = calinski_harabasz_score(genre_embendding_pca, prediction_genre)

# print(f"Silhouette: {s_score:.2f}")
# print(f"Daives Bouldin: {db_score:.2f}")
# print(f"Calinski Harabaz: {ch_score:.2f}")

ohe = OneHotEncoder(dtype=int)
ohe_columns = ohe.fit_transform(treated_music_data[["artists"]]).toarray()
treated_music_data_dummies = pd.concat([treated_music_data_without_artists, pd.DataFrame(ohe_columns, columns=ohe.get_feature_names_out(['artists']))], axis=1)

music_embendding_pca = pca_pipeline.fit_transform(treated_music_data_dummies)
music_projection = pd.DataFrame(data=music_embendding_pca)

model_music = KMeans(n_clusters=150)
model_music.fit(music_projection)
prediction_music = model_music.predict(music_projection)
treated_music_data['cluster'] = prediction_music
music_projection['cluster'] = prediction_music
music_projection['artist'] = treated_music_data['artists']
music_projection['song'] = raw_music_data['artists_song']

music_name = 'Alicia Keys - No One'

cluster =  list(music_projection[music_projection['song']== music_name]['cluster'])[0]
recomended_musics = music_projection[music_projection['cluster']== cluster][[0, 1, 'song']]
x_music = list(music_projection[music_projection['song']== music_name][0])[0]
y_music = list(music_projection[music_projection['song']== music_name][1])[0]

music_distances = euclidean_distances(recomended_musics[[0, 1]], [[x_music, y_music]])
recomended_musics['id'] = raw_music_data['id']
recomended_musics['distances'] = music_distances
recomended_musics['cluster'] = music_projection['cluster']

most_recomended_musics = recomended_musics.sort_values('distances')

scope = "user-library-read playlist-modify-private"
OAuth = SpotifyOAuth(
        scope=scope,         
        redirect_uri='http://localhost:5000/callback',
        client_id = '',
        client_secret = '')

client_credentials_manager = SpotifyClientCredentials(client_id = '',client_secret = '')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

id = raw_music_data[raw_music_data['artists_song'] == music_name]['id'].iloc[0]

name, url = recommend_id(recomended_musics['id'].head(10))
visualize_songs(name, url)
