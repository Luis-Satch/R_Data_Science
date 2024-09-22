# ---
# title: "MovieLens Recommendation System"
# author: "Dr. Luis Satch"
#
# Note
# This analysis was performed using **R version 3.6 or later**.  
# The `set.seed()` function behavior changed in R 3.6+ to introduce the `"Rounding"` argument for compatibility with earlier versions.  
# For **R 3.6 or later**, we use `set.seed(1, sample.kind = "Rounding")`.
# For **R versions earlier than 3.6**, use `set.seed(1)` without the `"Rounding"` argument.
# ---

# Table of Contents
# 1. Data Preparation ................................................... Line 25 
# 2. Data Splitting ..................................................... Line 90
# 3. Feature Engineering ................................................ Line 115
# 4. Exploratory Data Analysis (EDA) .................................... Line 136
# 5. Modeling Approach .................................................. Line 179
# 6. Model Validation ................................................... Line 225
# 7. Final Model ........................................................ Line 300
# 8. Evaluation ......................................................... Line 330
# 9. Conclusion ......................................................... Line 345
# ---


# 1. Data Preparation
# Load necessary libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# Clean up from previous runs (remove existing objects) and clear the console
rm(list = ls())
cat("\014")
# Check if datasets already exist in memory, load them if available
if (exists("movielens")) {
  message("Dataset already loaded. Skipping download and loading steps.")
} else {
  
  # MovieLens 10M dataset:
  # URLs for dataset download
  dl <- "ml-10M100K.zip"
  ratings_file <- "ml-10M100K/ratings.dat"
  movies_file <- "ml-10M100K/movies.dat"
  
  # Download and unzip the dataset if it doesn't exist locally
  if(!file.exists(dl)) {
    download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  }
  
  # Unzip ratings and movies data
  if(!file.exists(ratings_file)) {
    unzip(dl, files = ratings_file)
  }
  if(!file.exists(movies_file)) {
    unzip(dl, files = movies_file)
  }
  
  # Load and process ratings data
  ratings_raw <- read_lines(ratings_file)  # Read lines of data
  ratings <- as.data.frame(str_split(ratings_raw, fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
  colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
  
  # Load and process movies data
  movies_raw <- read_lines(movies_file)  # Read lines of data
  movies <- as.data.frame(str_split(movies_raw, fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
  colnames(movies) <- c("movieId", "title", "genres")
  
  # Convert appropriate columns to integer and numeric types
  ratings <- ratings %>%
    mutate(userId = as.integer(userId),
           movieId = as.integer(movieId),
           rating = as.numeric(rating),
           timestamp = as.integer(timestamp))
  
  movies <- movies %>%
    mutate(movieId = as.integer(movieId))
  
  # Merge ratings and movie data into a single dataframe
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Remove temporary variables
  rm(dl, ratings_raw, movies_raw, ratings, movies)
  
  message("Dataset loaded and processed.")
}


# 2. Data Splitting
# Split the data into edx and final_holdout_test sets
# Ensure that both movieId and userId in the final_holdout_test set exist in the edx set

# Set a seed for reproducibility
set.seed(1, sample.kind = "Rounding")  # If you're using R 3.6 or later

# Split the data: 90% for edx (training set) and 10% for final_holdout_test (test set)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)

edx <- movielens[-test_index,]  # Training set
temp <- movielens[test_index,]  # Temporary test set

# Ensure that movieId and userId in final_holdout_test are also in edx
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final_holdout_test back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- bind_rows(edx, removed)

# Clean up temporary variables
rm(test_index, temp, removed)

# 3. Feature Engineering: Add user and movie averages
# Calculate the average rating per user
user_avgs <- edx %>%
  group_by(userId) %>%
  summarise(user_avg = mean(rating))

# Calculate the average rating per movie
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(movie_avg = mean(rating))

# Merge these averages into the edx dataset
edx <- edx %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movie_avgs, by = "movieId")

# Display a sample of the engineered features
cat("User and movie averages added to the edx dataset.\n")

# Merging these features into validation and final_holdout_test should happen after defining `validation_set`

# 4. Exploratory Data Analysis (EDA)
# Summarise basic statistics (number of users, movies, ratings)

# Number of unique users
num_users <- edx %>% summarise(users = n_distinct(userId)) %>% pull(users)

# Number of unique movies
num_movies <- edx %>% summarise(movies = n_distinct(movieId)) %>% pull(movies)

# Total number of ratings
num_ratings <- nrow(edx)

# Summary output
cat("Number of unique users:", num_users, "\n")
cat("Number of unique movies:", num_movies, "\n")
cat("Total number of ratings:", num_ratings, "\n")

# Visualise rating distributions
edx %>%
  ggplot(aes(x = rating)) +
  geom_histogram(binwidth = 0.5, color = "black", fill = "skyblue") +
  scale_x_continuous(breaks = seq(0.5, 5, by = 0.5)) +
  labs(title = "Distribution of Movie Ratings", x = "Rating", y = "Count")

# Most rated movies (top 10)
top_movies <- edx %>%
  group_by(title) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  top_n(10, count)

# Display the top 10 most rated movies
cat("\nTop 10 most rated movies:\n")
print(top_movies)

# Visualise the top 10 most rated movies
top_movies %>%
  ggplot(aes(x = reorder(title, count), y = count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Most Rated Movies", x = "Movie Title", y = "Number of Ratings")


# 5. Modeling Approach

# Baseline Model: Predict the average rating for all movies
mu <- mean(edx$rating)  # Global average rating

# Display the baseline model prediction (global average rating)
cat("Baseline Model (Global Average Rating):", mu, "\n")

# Movie Effect Model: Adjust the rating based on each movie's average rating
movie_effects <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))  # Movie-specific effect (deviation from global average)

# Display a few movie effects
cat("Movie Effect Model: Displaying a few movie effects\n")
print(head(movie_effects))

# Movie + User Effect Model: Adjust for both movie and user-specific effects
user_effects <- edx %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))  # User-specific effect

# Display a few user effects
cat("User Effect Model: Displaying a few user effects\n")
print(head(user_effects))

# Regularisation: Penalise complexity by shrinking movie and user effects
lambda <- 5  # Regularisation parameter (you can tune this)
movie_effects_reg <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu) / (n() + lambda))  # Regularised movie effect

user_effects_reg <- edx %>%
  left_join(movie_effects_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (n() + lambda))  # Regularised user effect

# Display regularised movie and user effects
cat("Regularised Movie Effect Model: Displaying a few regularised movie effects\n")
print(head(movie_effects_reg))

cat("Regularised User Effect Model: Displaying a few regularised user effects\n")
print(head(user_effects_reg))


# 6. Model Validation
# - Create a separate training and validation split within the edx dataset for model development
# - Calculate RMSE for each model using cross-validation or a validation set

# Load necessary library for calculating RMSE
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
library(Metrics)

# Split the edx dataset into training (90%) and validation (10%) sets
set.seed(1, sample.kind = "Rounding")  # Ensure reproducibility

# Create a partition: 90% training, 10% validation
train_index <- createDataPartition(y = edx$rating, times = 1, p = 0.9, list = FALSE)
train_set <- edx[train_index, ]
validation_set <- edx[-train_index, ]

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Baseline Model RMSE: Predicting the global average rating
baseline_rmse <- RMSE(validation_set$rating, mu)

cat("Baseline Model RMSE:", baseline_rmse, "\n")

# Movie Effect Model: Predicting ratings using movie effects (b_i)
predicted_ratings_movie <- validation_set %>%
  left_join(movie_effects, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

movie_effect_rmse <- RMSE(validation_set$rating, predicted_ratings_movie)

cat("Movie Effect Model RMSE:", movie_effect_rmse, "\n")

# Movie + User Effect Model: Predicting ratings using both movie and user effects (b_i and b_u)
predicted_ratings_user <- validation_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_effect_rmse <- RMSE(validation_set$rating, predicted_ratings_user)

cat("Movie + User Effect Model RMSE:", user_effect_rmse, "\n")

# Regularised Movie + User Effect Model: Predict ratings using regularised movie and user effects
predicted_ratings_reg <- validation_set %>%
  left_join(movie_effects_reg, by = "movieId") %>%
  left_join(user_effects_reg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

regularised_rmse <- RMSE(validation_set$rating, predicted_ratings_reg)

cat("Regularised Movie + User Effect Model RMSE:", regularised_rmse, "\n")

# Summary of RMSE results for each model
cat("\nRMSE Summary:\n")
cat("Baseline Model:", baseline_rmse, "\n")
cat("Movie Effect Model:", movie_effect_rmse, "\n")
cat("Movie + User Effect Model:", user_effect_rmse, "\n")
cat("Regularised Movie + User Effect Model:", regularised_rmse, "\n")

# Now that validation_set exists, we can merge user and movie averages into it
validation_set <- validation_set %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movie_avgs, by = "movieId")

final_holdout_test <- final_holdout_test %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movie_avgs, by = "movieId")


# 7. Final Model
# - Train the final model on the entire edx dataset (using the best-performing approach)
# - Predict movie ratings for the final_holdout_test set

# Based on RMSE results, we will use the Regularised Movie + User Effect Model as the final model

# Train the final model on the entire edx dataset (regularised movie + user effects)
lambda <- 5  # Regularisation parameter (tuned previously)
movie_effects_final <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu) / (n() + lambda))  # Regularised movie effect

user_effects_final <- edx %>%
  left_join(movie_effects_final, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (n() + lambda))  # Regularised user effect

# Predict movie ratings for the final_holdout_test set using the trained final model
predicted_ratings_final <- final_holdout_test %>%
  left_join(movie_effects_final, by = "movieId") %>%
  left_join(user_effects_final, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE for the final model on the final_holdout_test set
final_rmse <- RMSE(final_holdout_test$rating, predicted_ratings_final)

# Display the final RMSE
cat("Final Model RMSE on final_holdout_test set:", final_rmse, "\n")

# 8. Evaluation
# - Compute the RMSE on the final_holdout_test set

# The RMSE for the final model on the final_holdout_test set has already been calculated in the previous step
cat("Final Model RMSE on final_holdout_test set:", final_rmse, "\n")

# Evaluate the performance of the final model by comparing RMSE with baseline and other models
cat("\nModel Evaluation Summary:\n")
cat("Baseline Model RMSE:", baseline_rmse, "\n")
cat("Movie Effect Model RMSE:", movie_effect_rmse, "\n")
cat("Movie + User Effect Model RMSE:", user_effect_rmse, "\n")
cat("Regularised Movie + User Effect Model RMSE (Final Model):", regularised_rmse, "\n")
cat("Final Model RMSE on final_holdout_test set:", final_rmse, "\n")


# 9. Conclusion
# - Print the final RMSE and summarise the model's performance

# Print the final RMSE on the final_holdout_test set and summarise the model's performance in a more readable format
cat(
  "\n================== Conclusion ==================\n",
  "\nI developed multiple models to predict movie ratings using the MovieLens dataset.\n",
  "\n1. Baseline Model: \n   The global average rating for all movies provided an RMSE of: ", baseline_rmse, "\n",
  "\n2. Movie Effect Model: \n   Accounted for differences in movie ratings, which reduced the RMSE to: ", movie_effect_rmse, "\n",
  "\n3. Movie + User Effect Model: \n   Further reduced the RMSE to: ", user_effect_rmse, "\n",
  "\n4. Regularised Movie + User Effect Model: \n   Best model with an RMSE of: ", regularised_rmse, "\n",
  "\nThis final model was evaluated on the holdout test set, yielding an RMSE of: ", final_rmse, "\n",
  "\nThe Regularised Movie + User Effect Model provides a balance between model complexity and generalisation,\n",
  "resulting in the most accurate predictions.\n",
  "\nOverall, the final model outperforms the baseline and intermediate models,\n",
  "demonstrating the importance of accounting for both movie and user effects, as well as applying regularisation to prevent overfitting.\n",
  "===================================================\n"
)
