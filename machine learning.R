# ==============================================================================
# MACHINE LEARNING & SPATIAL PREDICTION (R SCRIPT VERSION)
# Subtitle: Cross-Validation, Spatial Random Forests, and Geographic Feature Engineering
# Author: Neena Abimbola Albarus
# ==============================================================================

# ------------------------------------------------------------------------------
# SETUP & PACKAGES
# ------------------------------------------------------------------------------
library(tidyverse)
library(sf)
library(rnaturalearth)
library(ranger)
library(caret)
library(viridis)

# ------------------------------------------------------------------------------
# MODULE 1: DATA INGESTION & SPATIAL FEATURE ENGINEERING
# ------------------------------------------------------------------------------
# Dynamic path resolution across root or working directories
possible_paths <- c(
  "data/life_expectancy_data.csv",
  "../data/life_expectancy_data.csv",
  "/Users/yawdgyal/Desktop/R for github/Spatial Econometrics & Geographic Clustering/data/life_expectancy_data.csv"
)

data_path <- possible_paths[file.exists(possible_paths)][1]

if (is.na(data_path)) {
  stop("Could not locate dataset. Please check your working directory or file path.")
}

owid_raw <- read_csv(data_path)

# Fetch spatial boundaries from Natural Earth
world_polygons <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(continent != "Antarctica") %>%
  select(iso_a3, name, continent, geometry)

# Join attributes and extract centroids
ml_data <- world_polygons %>%
  inner_join(
    owid_raw %>% 
      filter(Year == 2019, !is.na(Code)) %>%
      rename(iso_a3 = Code, life_exp = `Life expectancy`),
    by = "iso_a3"
  ) %>%
  filter(!st_is_empty(geometry), !is.na(life_exp))

# Extract spatial coordinates (Longitude / Latitude centroids)
centroids <- st_centroid(ml_data)
coords <- st_coordinates(centroids)

ml_data <- ml_data %>%
  mutate(
    longitude = coords[, 1],
    latitude  = coords[, 2]
  )

print(head(ml_data %>% st_drop_geometry() %>% select(iso_a3, name, life_exp, longitude, latitude)))

# ------------------------------------------------------------------------------
# MODULE 2: STANDARD MACHINE LEARNING VS. SPATIAL LEAKAGE
# ------------------------------------------------------------------------------
# Prepare modeling dataset
df_model <- ml_data %>%
  st_drop_geometry() %>%
  select(life_exp, longitude, latitude, continent) %>%
  drop_na()

# Standard 5-Fold Cross-Validation (Non-Spatial)
set.seed(42)
train_control_random <- trainControl(method = "cv", number = 5)

rf_random_cv <- train(
  life_exp ~ .,
  data = df_model,
  method = "ranger",
  trControl = train_control_random,
  importance = "impurity"
)

print(rf_random_cv)

# ------------------------------------------------------------------------------
# MODULE 3: SPATIAL BLOCK CROSS-VALIDATION
# ------------------------------------------------------------------------------
# Create 5 spatial clusters (blocks) based on coordinates to prevent data leakage
set.seed(42)
spatial_clusters <- kmeans(df_model[, c("longitude", "latitude")], centers = 5)$cluster
df_model$spatial_block <- as.factor(spatial_clusters)

# Plot Spatial Blocks
fig1 <- ggplot(df_model, aes(x = longitude, y = latitude, color = spatial_block)) +
  geom_point(size = 3, alpha = 0.8) +
  scale_color_viridis_d(name = "Spatial Block") +
  theme_minimal() +
  labs(
    title = "Figure 1: Spatial Block CV Partitions",
    subtitle = "K-Means Geographic Clustering to Prevent Data Leakage"
  )

print(fig1)

# Evaluate out-of-block performance
spatial_folds <- unique(df_model$spatial_block)
metrics <- data.frame(fold = integer(), rmse = double(), r2 = double())

for (f in spatial_folds) {
  train_set <- df_model %>% filter(spatial_block != f)
  test_set  <- df_model %>% filter(spatial_block == f)
  
  fit <- ranger(
    life_exp ~ longitude + latitude + continent,
    data = train_set,
    importance = "impurity"
  )
  
  preds <- predict(fit, data = test_set)$predictions
  
  rmse_val <- sqrt(mean((test_set$life_exp - preds)^2))
  r2_val   <- cor(test_set$life_exp, preds)^2
  
  metrics <- rbind(metrics, data.frame(fold = f, rmse = rmse_val, r2 = r2_val))
}

spatial_cv_summary <- metrics %>%
  summarise(
    Mean_Spatial_RMSE = mean(rmse),
    Mean_Spatial_R2   = mean(r2, na.rm = TRUE)
  )

print(spatial_cv_summary)

# ------------------------------------------------------------------------------
# MODULE 4: VARIABLE IMPORTANCE DIAGNOSTICS
# ------------------------------------------------------------------------------
final_rf <- ranger(
  life_exp ~ longitude + latitude + continent,
  data = df_model,
  importance = "impurity"
)

imp_df <- data.frame(
  Feature = names(importance(final_rf)),
  Importance = importance(final_rf)
)

fig2 <- ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Figure 2: Random Forest Variable Importance",
    x = "Predictor Feature",
    y = "Node Impurity Reduction"
  )

print(fig2)

# ------------------------------------------------------------------------------
# STUDENT LAB EXERCISES
# ------------------------------------------------------------------------------

# --- Exercise 1: K=10 Spatial Blocks ---
set.seed(42)
df_model$block_10 <- as.factor(kmeans(df_model[, c("longitude", "latitude")], centers = 10)$cluster)

rmse_10 <- c()
for (f in unique(df_model$block_10)) {
  train_set <- df_model %>% filter(block_10 != f)
  test_set  <- df_model %>% filter(block_10 == f)
  
  fit <- ranger(life_exp ~ longitude + latitude + continent, data = train_set)
  preds <- predict(fit, data = test_set)$predictions
  rmse_10 <- c(rmse_10, sqrt(mean((test_set$life_exp - preds)^2)))
}

cat("Exercise 1 - Mean Spatial RMSE (K=10):", mean(rmse_10, na.rm = TRUE), "\n")

# --- Exercise 2: Equator Distance Feature ---
df_model <- df_model %>% mutate(abs_lat = abs(latitude))

rmse_feat <- c()
for (f in unique(df_model$spatial_block)) {
  train_set <- df_model %>% filter(spatial_block != f)
  test_set  <- df_model %>% filter(spatial_block == f)
  
  fit <- ranger(life_exp ~ longitude + latitude + continent + abs_lat, data = train_set)
  preds <- predict(fit, data = test_set)$predictions
  rmse_feat <- c(rmse_feat, sqrt(mean((test_set$life_exp - preds)^2)))
}

cat("Exercise 2 - Mean Spatial RMSE with Equator Distance:", mean(rmse_feat, na.rm = TRUE), "\n")