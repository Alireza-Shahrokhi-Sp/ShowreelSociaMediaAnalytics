rm(list=ls())
library(settings)
reset(options)
graphics.off()

library(nlmeU)  # --> for the dataset
library(nlme)   # --> for models implementation
library(lattice)
library(corrplot)
library(plot.matrix)
library(visdat)
library(car)
library(ggplot2)
library(lme4)
library(insight)
library(RLRsim) 
library(dplyr)
library(lubridate)
library(forcats)
library(caret)
library(randomForest)
library(GGally)
library(tidyr)
library(glmmTMB)
library(DHARMa)

df <- read.csv("C:/Users/miche/Desktop/A4B LAB/Project A4B/Cleaned data/Ig Single cluster analysis/Activation models/inactive_dataset.csv", 
               colClasses = c(post_id = "character"))            
head(df)
df$post_id <- as.factor(df$post_id)
df$is_weekend <- as.logical(df$is_weekend)
df$hour_band <- as.factor(df$hour_band)
df$media_type <- as.factor(df$media_type)

df$mentions_alice <- as.logical(df$mentions_alice)
df$mentions_guglielmo <- as.logical(df$mentions_guglielmo)
df$mentions_aimone <- as.logical(df$mentions_aimone)
df$mentions_claudio <- as.logical(df$mentions_claudio)
df$mentions_creators <- as.logical(df$mentions_creators)
df$mentions_famous <- as.logical(df$mentions_famous)
df$mentions_singer <- as.logical(df$mentions_singer)
df$mentions_fashion <- as.logical(df$mentions_fashion)
df$has_mentions <- as.logical(df$has_mentions)
df$mentions_brand <- as.logical(df$mentions_brand)
df$mentions_other_people <- as.logical(df$mentions_other_people)
df$is_adv <- as.logical(df$is_adv)
df$is_supplied <- as.logical(df$is_supplied)
df$is_gifted <- as.logical(df$is_gifted)
df$is_sanremo <- as.logical(df$is_sanremo)
df$topic <- as.factor(df$topic)
df$has_love_emoji <- as.logical(df$has_love_emoji)
df$has_lol_emoji <- as.logical(df$has_lol_emoji)
df$has_shine_emoji <- as.logical(df$has_shine_emoji)
df$has_call_to_action <- as.logical(df$has_call_to_action)
df$is_ironic <- as.logical(df$is_ironic)
df$audio_type <- as.factor(df$audio_type)
df$location <- as.factor(df$location)

df$is_video <- ifelse(is.na(df$video_duration), 0, 1)

df$video_duration_filled <- ifelse(
  is.na(df$video_duration),
  0,
  df$video_duration
)

df$n_hashtags[is.na(df$n_hashtags)] <- 0
df$n_mentions[is.na(df$n_mentions)] <- 0

df <- df %>%
  mutate(
    section = case_when(
      media_type %in% c("CAROUSEL", "IMAGE") ~ "FEED",
      media_type == "VIDEO_REELS" ~ "REELS",
      TRUE ~ NA_character_
    )
  )

df$section <- as.factor(df$section)


df <- df %>%
  mutate(
    has_close_friend = ifelse(
      mentions_alice | mentions_guglielmo | mentions_claudio,
      1, 0
    )
  )

# Create year factor
df <- df %>%
  mutate(
    year_f = factor(
      format(as.POSIXct(as.character(timestamp), format = "%Y-%m-%d %H:%M:%S"), "%Y"),
      levels = c("2023", "2024", "2025", "2026")
    )
  )

# Create quadrimester factor
ts <- as.POSIXct(as.character(df$timestamp), format = "%Y-%m-%d %H:%M:%S")
yr <- format(ts, "%Y")
mn <- as.integer(format(ts, "%m"))

quad <- ifelse(mn <= 4, "t1",
               ifelse(mn <= 8, "t2", "t3"))

df$quadrimester_f <- paste(quad, yr)

# Set quadrimester factor levels in chronological order
df$quadrimester_f <- factor(
  df$quadrimester_f,
  levels = c(
    "t1 2023", "t2 2023", "t3 2023",
    "t1 2024", "t2 2024", "t3 2024",
    "t1 2025", "t2 2025", "t3 2025", "t1 2026"
  )
)



df <- df %>%
  mutate(
    season = case_when(
      month %in% c(12, 1, 2) ~ "winter",
      month %in% c(3, 4, 5) ~ "spring",
      month %in% c(6, 7, 8) ~ "summer",
      month %in% c(9, 10, 11) ~ "autumn",
      TRUE ~ NA_character_
    ),
    
    season_year = case_when(
      month == 12 ~ year + 1,
      TRUE ~ year
    ),
    
    season = factor(
      season,
      levels = c("winter","spring", "summer", "autumn" )
    )
  )


df <- df %>%
  arrange(timestamp) %>%
  mutate(post_number = row_number())

df$hour <- as.factor(df$hour)
df$month <- as.factor(df$month)

library(fastDummies)

df_rf<- df %>%
  dummy_cols(
    select_columns = "topic",
    remove_first_dummy = FALSE,
    remove_selected_columns = TRUE
  )

topic_dummy_vars <- grep("^topic_", names(df_rf), value = TRUE)

df_rf[topic_dummy_vars] <- lapply(df_rf[topic_dummy_vars], as.logical)
names(df_rf) <- gsub("&", "and", names(df_rf))
names(df_rf) <- gsub("[^A-Za-z0-9_]", "_", names(df_rf))
names(df_rf) <- gsub("_+", "_", names(df_rf))
names(df_rf) <- gsub("_$", "", names(df_rf))

##### Train-test split ####
set.seed(123)
train_idx <- createDataPartition(df_rf$share_inactive_pre, p = 0.7, list = FALSE)

train_df <- df_rf[train_idx, ]
test_df  <- df_rf[-train_idx, ]

# Define formula (target ~ features)
rf_formula <- share_inactive_pre ~ year + month + is_weekend + hour_band + hour + media_type+
  mentions_alice+ mentions_guglielmo+ mentions_aimone+ mentions_claudio +
  mentions_famous+ mentions_creators + mentions_singer + mentions_fashion+
  mentions_brand + mentions_other_people + is_adv + 
  is_supplied + is_gifted + is_sanremo +caption_length +avg_words_per_sentence+
  exclamation_marks + question_marks + has_love_emoji + has_lol_emoji + has_shine_emoji+
  has_call_to_action + posting_frequency_last7d + days_since_last_post + days_since_last_adv +
  adv_density_last30d + same_topic_last_10posts+ is_ironic+ audio_type + location+
  n_hashtags  + video_duration_filled +section +season +topic_Beauty_and_Fashion + topic_Cinema_and_TV+
  topic_Community + topic_Cooking + topic_Culture + topic_Food + topic_Games +
  topic_Other + topic_Podcast + topic_Private_Life +
  topic_Reflections + topic_Relatable_Comedy + topic_Theatre +
  topic_Travel + topic_YouTube

# Train RF
set.seed(123)
rf_model <- randomForest(
  formula = rf_formula,
  data    = train_df,
  ntree   = 300,
  mtry    = 4,
  importance = TRUE
)

print(rf_model)

# Show importance table
importance(rf_model)

# Plot variable importance
varImpPlot(
  rf_model,
  main = "Random Forest - Variable Importance",
  n.var = 30,            # show top 
  type = 1               # type=1 → mean decrease in accuracy
)

#### Beta-binomial model ####

library(glmmTMB)
train_df <- train_df %>%
  mutate(
    year = factor(year),
    month = factor(month),
    season = factor(season),
    hour_band = factor(hour_band),
    hour = factor(hour), 
    location = factor(location),
    section = factor(section),
    audio_type = factor(audio_type)
  ) %>%
  filter(n_total_reactivation > 0)

test_df <- test_df %>%
  mutate(
    year = factor(year),
    month = factor(month),
    season = factor(season),
    hour_band = factor(hour_band),
    hour = factor(hour),
    location = factor(location),
    section = factor(section),
    audio_type = factor(audio_type)
  ) %>%
  filter(n_total_reactivation > 0)

mod_full <- glmmTMB(
  cbind(n_inactive_pre, n_active_pre) ~
    year +
    season +
    video_duration_filled +
    mentions_other_people+
    section +
    avg_words_per_sentence +
    topic_Relatable_Comedy +
    topic_Cinema_and_TV, 
  family = betabinomial(),
  data = train_df
)

summary(mod_full)

mod_final <- mod_full

# Check for residuals
set.seed(123)

res_final <- simulateResiduals(
  fittedModel = mod_final,
  n = 1000
)

plot(res_final)

testUniformity(res_final)
testDispersion(res_final)
testZeroInflation(res_final)
testOutliers(res_final)

library(performance)
check_collinearity(mod_final)


train_df$share_obs <- train_df$n_inactive_pre / 
  (train_df$n_inactive_pre + train_df$n_active_pre)

train_df$share_pred <- predict(mod_final, type = "response")

ggplot(train_df, aes(x = share_pred, y = share_obs)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Observed vs predicted not yet active share",
    x = "Predicted share not yet active",
    y = "Observed share not yet active"
  )

# Zero inflation
mod_inactive_zi <- glmmTMB(
  cbind(n_inactive_pre, n_active_pre) ~
    year +
    season +
    audio_type +
    mentions_singer +
    has_call_to_action +
    mentions_fashion,
  ziformula = ~ 1,
  family = betabinomial(),
  data = train_df
)

summary(mod_inactive_zi)

AIC(mod_final, mod_inactive_zi)
BIC(mod_final, mod_inactive_zi)

mod_inactive_zi_reduced <- glmmTMB(
  cbind(n_inactive_pre, n_active_pre) ~
    year +
    season +
    audio_type +
    mentions_singer,
  ziformula = ~ 1,
  family = betabinomial(),
  data = train_df
)

mod_inactive_zi_core <- glmmTMB(
  cbind(n_inactive_pre, n_active_pre) ~
    year +
    season +
    audio_type,
  ziformula = ~ 1,
  family = betabinomial(),
  data = train_df
)

AIC(
  mod_inactive_zi,
  mod_inactive_zi_reduced,
  mod_inactive_zi_core
)

BIC(
  mod_inactive_zi,
  mod_inactive_zi_reduced,
  mod_inactive_zi_core
)

models_inactive <- list(
  inactive_zi_full = mod_inactive_zi,
  inactive_zi_reduced = mod_inactive_zi_reduced,
  inactive_zi_core = mod_inactive_zi_core
)

inactive_model_compare <- data.frame(
  model = c("full", "reduced", "core"),
  df = c(
    attr(logLik(mod_inactive_zi), "df"),
    attr(logLik(mod_inactive_zi_reduced), "df"),
    attr(logLik(mod_inactive_zi_core), "df")
  ),
  AIC = c(
    AIC(mod_inactive_zi),
    AIC(mod_inactive_zi_reduced),
    AIC(mod_inactive_zi_core)
  ),
  BIC = c(
    BIC(mod_inactive_zi),
    BIC(mod_inactive_zi_reduced),
    BIC(mod_inactive_zi_core)
  )
) %>%
  mutate(
    delta_AIC = AIC - min(AIC),
    delta_BIC = BIC - min(BIC)
  ) %>%
  arrange(BIC)

inactive_model_compare

#### test df correction ####
library(dplyr)
library(purrr)

# ------------------------------------------------------------
# 1. Align factor levels between train_df and test_df
# ------------------------------------------------------------

fix_to_train_levels <- function(x, train_x) {
  
  train_levels <- levels(factor(train_x))
  x_chr <- trimws(as.character(x))
  x_low <- tolower(x_chr)
  
  # Boolean TRUE/FALSE variables
  if (all(train_levels %in% c("FALSE", "TRUE"))) {
    x_chr <- case_when(
      x_low %in% c("true", "t", "1", "yes", "y", "si", "sì") ~ "TRUE",
      x_low %in% c("false", "f", "0", "no", "n") ~ "FALSE",
      TRUE ~ NA_character_
    )
  }
  
  factor(x_chr, levels = train_levels)
}

factor_vars <- c(
  "year",
  "season",
  "audio_type",
  "mentions_singer",
  "has_call_to_action",
  "mentions_fashion"
)

for (v in factor_vars) {
  if (v %in% names(train_df) && v %in% names(test_df)) {
    train_df[[v]] <- factor(train_df[[v]])
    test_df[[v]] <- fix_to_train_levels(test_df[[v]], train_df[[v]])
  }
}

sapply(test_df[factor_vars], function(x) sum(is.na(x)))

get_rhs_formula <- function(model) {
  f <- formula(model)
  rhs <- paste(deparse(f[[3]]), collapse = " ")
  as.formula(paste("~", rhs))
}

manual_predict_glmmTMB <- function(model, newdata, include_zi = TRUE) {
  
  # Conditional fixed effects
  beta_cond <- fixef(model)$cond
  
  rhs_formula <- get_rhs_formula(model)
  X <- model.matrix(rhs_formula, data = newdata)
  
  # Add missing columns as zero
  missing_cols <- setdiff(names(beta_cond), colnames(X))
  
  if (length(missing_cols) > 0) {
    for (mc in missing_cols) {
      X <- cbind(X, setNames(data.frame(0), mc))
    }
  }
  
  # Drop extra columns and reorder
  X <- X[, names(beta_cond), drop = FALSE]
  
  eta <- as.vector(X %*% beta_cond)
  mu <- plogis(eta)
  
  # Zero-inflation component, only for ziformula = ~ 1
  beta_zi <- fixef(model)$zi
  
  if (include_zi && length(beta_zi) > 0) {
    if (length(beta_zi) == 1 && "(Intercept)" %in% names(beta_zi)) {
      p_zero <- plogis(unname(beta_zi["(Intercept)"]))
      mu <- (1 - p_zero) * mu
    } else {
      stop("Questa funzione gestisce per ora solo ziformula = ~ 1.")
    }
  }
  
  mu
}

#### test df ####
eval_inactive_model <- function(model, test_df, model_name) {
  
  predictors <- setdiff(
    all.vars(formula(model)),
    c("n_inactive_pre", "n_active_pre")
  )
  
  df_eval <- test_df %>%
    mutate(
      n_total_inactive = n_inactive_pre + n_active_pre,
      obs_inactive_share = n_inactive_pre / n_total_inactive
    ) %>%
    filter(n_total_inactive > 0) %>%
    filter(if_all(all_of(predictors), ~ !is.na(.)))
  
  pred <- manual_predict_glmmTMB(
    model = model,
    newdata = df_eval,
    include_zi = TRUE
  )
  
  pred <- pmin(pmax(pred, 1e-6), 1 - 1e-6)
  
  obs <- df_eval$obs_inactive_share
  w <- df_eval$n_total_inactive
  k <- df_eval$n_inactive_pre
  n <- df_eval$n_total_inactive
  
  weighted_r2 <- 1 -
    sum(w * (obs - pred)^2, na.rm = TRUE) /
    sum(w * (obs - weighted.mean(obs, w, na.rm = TRUE))^2, na.rm = TRUE)
  
  binom_logloss <- -sum(
    k * log(pred) + (n - k) * log(1 - pred),
    na.rm = TRUE
  ) / sum(n, na.rm = TRUE)
  
  data.frame(
    model = model_name,
    n_test_posts = nrow(df_eval),
    RMSE_weighted = sqrt(weighted.mean((obs - pred)^2, w, na.rm = TRUE)),
    MAE_weighted = weighted.mean(abs(obs - pred), w, na.rm = TRUE),
    Bias_weighted = weighted.mean(obs - pred, w, na.rm = TRUE),
    Weighted_R2 = weighted_r2,
    Binomial_LogLoss = binom_logloss
  )
}
models_inactive <- list(
  inactive_zi_full = mod_inactive_zi,
  inactive_zi_reduced = mod_inactive_zi_reduced,
  inactive_zi_core = mod_inactive_zi_core
)

test_results_inactive <- purrr::imap_dfr(
  models_inactive,
  ~ eval_inactive_model(.x, test_df, .y)
) %>%
  arrange(Binomial_LogLoss)

test_results_inactive

#### Diagnostics ####
set.seed(123)

res_inactive_reduced <- simulateResiduals(
  fittedModel = mod_inactive_zi_reduced,
  n = 1000
)

plot(res_inactive_reduced)

testUniformity(res_inactive_reduced)
testDispersion(res_inactive_reduced)
testZeroInflation(res_inactive_reduced)
testOutliers(res_inactive_reduced)

plotResiduals(res_inactive_reduced, form = train_df$year)
plotResiduals(res_inactive_reduced, form = train_df$season)
plotResiduals(res_inactive_reduced, form = train_df$audio_type)

# Calibration plot
test_inactive_calib <- test_df %>%
  mutate(
    n_total_inactive = n_inactive_pre + n_active_pre,
    obs_inactive_share = n_inactive_pre / n_total_inactive
  ) %>%
  filter(n_total_inactive > 0)

test_inactive_calib$pred_inactive <- manual_predict_glmmTMB(
  model = mod_inactive_zi_reduced,
  newdata = test_inactive_calib,
  include_zi = TRUE
)

test_inactive_calib <- test_inactive_calib %>%
  mutate(
    pred_inactive = pmin(pmax(pred_inactive, 1e-6), 1 - 1e-6),
    pred_bin = ntile(pred_inactive, 5)
  ) %>%
  group_by(pred_bin) %>%
  summarise(
    mean_pred = weighted.mean(pred_inactive, n_total_inactive, na.rm = TRUE),
    obs_rate = sum(n_inactive_pre, na.rm = TRUE) /
      sum(n_total_inactive, na.rm = TRUE),
    n_posts = n(),
    n_users = sum(n_total_inactive, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(test_inactive_calib, aes(x = mean_pred, y = obs_rate, size = n_users)) +
  geom_point() +
  geom_text(aes(label = paste0("n=", n_posts)), vjust = -1, size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    x = "Mean predicted inactive share",
    y = "Observed inactive share",
    size = "Known commenters",
    title = "Calibration plot for inactive reactivation model",
    subtitle = "Test set"
  ) +
  theme_minimal()

mod_inactive_zi_core_disp_year <- glmmTMB(
  cbind(n_inactive_pre, n_active_pre) ~
    year + season + audio_type,
  ziformula = ~ 1,
  dispformula = ~ year,
  family = betabinomial(),
  data = train_df
)

mod_inactive_zi_core_disp_season <- glmmTMB(
  cbind(n_inactive_pre, n_active_pre) ~
    year + season + audio_type,
  ziformula = ~ 1,
  dispformula = ~ season,
  family = betabinomial(),
  data = train_df
)

mod_inactive_zi_core_disp_year_season <- glmmTMB(
  cbind(n_inactive_pre, n_active_pre) ~
    year + season + audio_type,
  ziformula = ~ 1,
  dispformula = ~ year + season,
  family = betabinomial(),
  data = train_df
)

AIC(
  mod_inactive_zi_core,
  mod_inactive_zi_core_disp_year,
  mod_inactive_zi_core_disp_season,
  mod_inactive_zi_core_disp_year_season
)

BIC(
  mod_inactive_zi_core,
  mod_inactive_zi_core_disp_year,
  mod_inactive_zi_core_disp_season,
  mod_inactive_zi_core_disp_year_season
)
