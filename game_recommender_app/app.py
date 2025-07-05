import json
from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)


# --- Data and Model Loading ---
def load_all_data():
    """Loads all necessary data files and models from their respective folders."""
    try:
        # Load models from the /models directory
        nn_model = load('models/nearest_neighbors_model.joblib')
        game_names_list = load('models/game_names_list.joblib')

        # Load data from the /data directory
        with open('data/game_to_cluster_map.json', 'r') as f:
            game_to_cluster = json.load(f)
        with open('data/cluster_descriptions.json', 'r') as f:
            long_descriptions = json.load(f)
        with open('data/cluster_short_descriptions.json', 'r') as f:
            short_descriptions = json.load(f)
        # Load the new file with games sorted by rank
        with open('data/top_10_games_by_rank_per_cluster.json', 'r') as f:
            top_10_games_by_rank = json.load(f)

        # Load the pre-scaled data for the NN model
        df_scaled = pd.read_csv('data/X_scaled_data.csv')

        return nn_model, game_to_cluster, long_descriptions, short_descriptions, top_10_games_by_rank, game_names_list, df_scaled
    except FileNotFoundError as e:
        print(f"Critical Error: Data or model file not found: {e.filename}")
        print("Please ensure all files are in their correct /data and /models directories.")
        return None, None, None, None, None, None, None


# Load everything on startup
nn_model, game_to_cluster, long_descriptions, short_descriptions, top_10_games_by_rank, game_names_list, df_scaled = load_all_data()

# Prepare data for processing if loading was successful
if df_scaled is not None:
    all_game_names = sorted(list(game_to_cluster.keys()))
    # Convert the pre-scaled DataFrame to a NumPy array for efficient lookups
    X_scaled_vectors = df_scaled.values
else:
    all_game_names = []
    X_scaled_vectors = None


@app.route('/', methods=['GET', 'POST'])
def index():
    selected_game = None
    description = None
    top_games_list = None
    similar_games_list = None
    error = None
    description_type = 'long'

    if request.method == 'POST':
        selected_game = request.form.get('game_name')
        description_type = request.form.get('desc_type', 'long')

        if selected_game and selected_game in game_to_cluster:
            # --- Group Description & Top Games Logic ---
            cluster_id = game_to_cluster.get(selected_game)
            if description_type == 'short':
                description = short_descriptions.get(str(cluster_id))
            else:
                description = long_descriptions.get(str(cluster_id))

            # Use the new dictionary with games sorted by rank
            top_games_list = top_10_games_by_rank.get(f'cluster_{cluster_id}')

            # --- Similar Games Logic (Nearest Neighbors) ---
            try:
                game_idx = game_names_list.index(selected_game)
                game_vector = X_scaled_vectors[game_idx]
                distances, indices = nn_model.kneighbors([game_vector])
                similar_indices = indices[0][1:]
                similar_games_list = [game_names_list[i] for i in similar_indices]
            except (ValueError, IndexError):
                similar_games_list = []
                print(f"Warning: Game '{selected_game}' found in cluster map but could not be found for NN lookup.")

        elif selected_game:
            error = f"Game '{selected_game}' was not found in our database. Please try selecting a game from the list."

    return render_template(
        'index.html',
        game_names=all_game_names,
        selected_game=selected_game,
        description=description,
        top_games=top_games_list,
        similar_games=similar_games_list,
        error=error,
        description_type=description_type
    )


if __name__ == '__main__':
    if nn_model is None or df_scaled is None:
        print("Application cannot start due to missing models or data files.")
    else:
        app.run(debug=True)