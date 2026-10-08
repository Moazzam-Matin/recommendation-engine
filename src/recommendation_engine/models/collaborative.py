import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCollaborativeFilter:
    """ Item-Based Collaboratve filtering recommendation model."""

    def __init__(self, n_neighbours=50, min_co_ratings=5):
        if not isinstance(n_neighbours, int) or isinstance(n_neighbours, bool):
            raise TypeError("n_neighbours must be an integer.")

        if n_neighbours <= 0:
            raise ValueError("n_neighbours must be greater than 0.")

        if not isinstance(min_co_ratings, int) or isinstance(min_co_ratings, bool):
            raise TypeError("min_co_ratins must be an integer")

        if min_co_ratings <= 0:
            raise ValueError("min_co_ratins must be greater than 0.")

        self.n_neighbours = n_neighbours
        self.min_co_ratings = min_co_ratings

        self._is_fitted = False


    def fit(self, ratings):
        """Validate and prepare rating data for model training.
        And fit the Item-Based Collaborative Filtering model."""

        if not isinstance(ratings, pd.DataFrame):
            raise TypeError("ratings must be a pandas DataFrame.")

        required_columns = {"UserID", "MovieID", "Rating"}
        missing_columns = required_columns - set(ratings.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"ratings is missing required columns: {missing}")

        if ratings.empty:
            raise ValueError("ratings must not be empty.")

        if ratings[list(required_columns)].isnull().any().any():
            raise ValueError("ratings must not contain missing values in UserID, MovieID and Rating.")

        if not pd.api.types.is_numeric_dtype(ratings["Rating"]):
            raise TypeError("ratings must be numeric.")

        ratings = ratings[["UserID", "MovieID", "Rating"]].copy()

        ratings = ratings.groupby(["UserID", "MovieID"], as_index=False)["Rating"].mean()

        movie_ids = ratings["MovieID"].drop_duplicates().tolist()
        user_ids = ratings["UserID"].drop_duplicates().tolist()

        movie_to_index = {
            movie_id: index
            for index, movie_id in enumerate(movie_ids)
        }

        user_to_index = {
            user_id: index
            for index, user_id in enumerate(user_ids)
        }

        rows = ratings["MovieID"].map(movie_to_index).to_numpy()
        cols = ratings["UserID"].map(user_to_index).to_numpy()
        values = ratings["Rating"].to_numpy()

        item_user_matrix = csr_matrix(
            (values, (rows, cols)),
            shape = (len(movie_ids), len(user_ids)),
        )

        self._ratings = ratings
        self._movie_to_index = movie_to_index
        self._user_to_index = user_to_index
        self._index_to_movie = {
            index: movie_id
            for movie_id, index in movie_to_index.items()
        }
        self._item_user_matrix = item_user_matrix

        self._compute_similarity()

        self._is_fitted = True

        return self

    def _compute_similarity(self):
        """"Compute and store the top similar movies for each movie."""

        similarity_matrix = cosine_similarity(self._item_user_matrix)

        #Count users who rated each pair of movies.
        interaction_matrix = self._item_user_matrix.copy()
        interaction_matrix.data = (interaction_matrix.data !=0).astype(int)

        co_rating_counts = interaction_matrix @ interaction_matrix.T

        self._similar_items = {}

        for movie_index in range(similarity_matrix.shape[0]):
            similarities = similarity_matrix[movie_index]
            co_ratings = co_rating_counts.getrow(movie_index).toarray().ravel()

            valid_indices = [
                index
                for index in range(len(similarities))
                if (
                    index != movie_index
                    and co_ratings[index] >= self.min_co_ratings
                    and similarities[index] > 0
                )
            ]
            valid_indices.sort(
                key=lambda index: similarities[index],
                reverse=True
            )

            top_indices = valid_indices[: self.n_neighbours]

            movie_id = self._index_to_movie[movie_index]

            self._similar_items[movie_id] = [
                (
                self._index_to_movie[index],
                float(similarities[index]),
                )
                for index in top_indices
            ]

    def recommend(self, user_id, n=10):
        """Generate top-N movie recommendations for a user."""

        if not self._is_fitted:
            raise NotFittedError("The model must be fitted before generating recommendations.")

        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("n must be an integer.")

        if n <= 0:
            raise ValueError("n must be greater than 0.")

        if user_id not in self._user_to_index:
            raise ValueError(f"Unknown user id: {user_id}")

        user_ratings = self._ratings[
            self._ratings["UserID"] == user_id
        ]
        rated_movie_ids = set(user_ratings["MovieID"])

        candidate_scores = {}

        for _, row in user_ratings.iterrows():
            rated_movie_id = row["MovieID"]
            rating = row["Rating"]
            similar_movies = self._similar_items.get(rated_movie_id, [])

            for candidate_movie_id, similarity in similar_movies:
                if candidate_movie_id in rated_movie_ids:
                    continue
                weighted_rating = rating * similarity

                candidate_scores.setdefault(candidate_movie_id, [0.0, 0.0])

                candidate_scores[candidate_movie_id][0] += weighted_rating
                candidate_scores[candidate_movie_id][1] += similarity

        recommendation_scores = {}

        for candidate_movie_id, (weighted_sum, similarity_sum) in candidate_scores.items():
            recommendation_scores[candidate_movie_id] = weighted_sum / similarity_sum

        ranked_candidates = sorted(
            recommendation_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return [movie_id for movie_id, _ in ranked_candidates[:n]]






            

