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

df <- read.csv("C:/Users/miche/Desktop/A4B LAB/Project A4B/Cleaned data/Ig Single cluster analysis/Activation models/activation_descriptive_analysis.csv", 
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
train_idx <- createDataPartition(df_rf$share_not_yet_active_pre, p = 0.7, list = FALSE)

train_df <- df_rf[train_idx, ]
test_df  <- df_rf[-train_idx, ]

# Define formula (target ~ features)
rf_formula <- share_not_yet_active_pre ~ year + month + is_weekend + hour_band + hour + media_type+
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

#### Correlation ####

# Assicurati che siano binarie 0/1
df$mentions_singer_num <- as.integer(df$mentions_singer == TRUE)
df$is_sanremo_num <- as.integer(df$is_sanremo == TRUE)

# Correlazione phi
cor(
  df$mentions_singer_num,
  df$is_sanremo_num,
  use = "complete.obs",
  method = "pearson"
)

cor.test(
  df$mentions_singer_num,
  df$is_sanremo_num,
  use = "complete.obs",
  method = "pearson"
)

#### Plots ####

# Histogram of Not_yet_active percentage 
ggplot(df, aes(x = share_not_yet_active_pre)) +
  geom_histogram(bins = 50, alpha = 0.6) +
  labs(
    title = "Not_yet_active_share_prev Histogram"
  ) +
  theme_minimal()

# Histogram of Not_yet_active percentage by media type
ggplot(df, aes(x = share_not_yet_active_pre, fill = section)) +
  geom_histogram(bins = 50, alpha = 0.6) +
  facet_wrap(~section) +
  theme_minimal()


# Boxplot of Not_yet_active percentage by section

ggplot(df, aes(share_not_yet_active_pre, fill = section)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Not_yet_active_prev % by section")

###### Friends boxplots ######
# Boxplot of Not_yet_active percentage by has_close_friend
ggplot(df, aes(
  y = factor(has_close_friend, labels = c("No", "Yes")),
  x = share_not_yet_active_pre,
  fill = factor(has_close_friend, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by has_close_friend",
    y = "Has Close friend",
    x = "Not_yet_active_prev percentage",
    fill = "Has close friend"
  )

# Boxplot of Not_yet_active percentage by mentions_guglielmo
ggplot(df, aes(
  y = factor(mentions_guglielmo, labels = c("No", "Yes")),
  x = share_not_yet_active_pre,
  fill = factor(mentions_guglielmo, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by mentions_guglielmo",
    y = "Mentions Guglielmo",
    x = "Not_yet_active_prev percentage",
    fill = "Mentions Guglielmo"
  )

# Boxplot of Not_yet_active percentage by mentions_alice
ggplot(df, aes(
  y = factor(mentions_alice, labels = c("No", "Yes")),
  x = share_not_yet_active_pre,
  fill = factor(mentions_alice, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by mentions_alice",
    y = "Mentions Alice",
    x = "Not_yet_active_prev percentage",
    fill = "Mentions Alice"
  )

# Boxplot of Not_yet_active percentage by mentions_fashion
ggplot(df, aes(
  y = factor(mentions_fashion, labels = c("No", "Yes")),
  x = share_not_yet_active_pre,
  fill = factor(mentions_fashion, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by mentions_fashion",
    y = "Mentions fashion",
    x = "Not_yet_active_prev percentage",
    fill = "Mentions fashion"
  )

# Boxplot of Not_yet_active percentage by mentions_aimone
ggplot(df, aes(
  y = factor(mentions_aimone, labels = c("No", "Yes")),
  x = share_not_yet_active_pre,
  fill = factor(mentions_aimone, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by mentions_aimone",
    y = "Mentions aimone",
    x = "Not_yet_active_prev percentage",
    fill = "Mentions aimone"
  )

###### Boxplot of posts characteristics ######

# Boxplot of Not_yet_active percentage by is_adv
ggplot(df, aes(
  y = factor(is_adv, labels = c("No", "Yes")),
  x = share_not_yet_active_pre,
  fill = factor(is_adv, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by is_adv",
    y = "Is_adv",
    x = "Not_yet_active_prev percentage",
    fill = "Is_adv"
  )

# Boxplot of Not_yet_active percentage by is_sanremo
ggplot(df, aes(
  y = factor(is_sanremo, labels = c("No", "Yes")),
  x = share_not_yet_active_pre,
  fill = factor(is_sanremo, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by is_sanremo",
    y = "Is_sanremo",
    x = "Not_yet_active_prev percentage",
    fill = "Is_sanremo"
  )

# Boxplot of Not_yet_active percentage by topic (excluding Uncertain, Dance and Other)
ggplot(
  subset(df, !topic %in% c("Uncertain", "Dance", "Other")),
  aes(share_not_yet_active_pre, fill = topic)
) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Not_yet_active_prev % by topic")

# Boxplot of Not_yet_active percentage by location
ggplot(df, aes(
  y = factor(location, labels = c("fake", "no", "real")),
  x = share_not_yet_active_pre,
  fill = factor(location, labels = c("fake", "no", "real"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by location",
    y = "Location",
    x = "Not_yet_active_prev percentage",
    fill = "Location"
  )

# Boxplot of Not_yet_active percentage by audio_type
ggplot(df, aes(
  y = factor(audio_type, labels = c("licensed", "original", "no")),
  x = share_not_yet_active_pre,
  fill = factor(audio_type, labels = c("licensed", "original", "no"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by audio_type",
    y = "Audio type",
    x = "Not_yet_active_prev percentage",
    fill = "audio type"
  )

# Boxplot of Not_yet_active percentage by has_call_to_action
ggplot(df, aes(
  y = factor(has_call_to_action),
  x = share_not_yet_active_pre,
  fill = factor(has_call_to_action)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by has_call_to_action",
    y = "has_call_to_action",
    x = "Not_yet_active_prev percentage",
    fill = "has_call_to_action"
  )

# Boxplot of Not_yet_active percentage by has_lol_emoji
ggplot(df, aes(
  y = factor(has_lol_emoji),
  x = share_not_yet_active_pre,
  fill = factor(has_lol_emoji)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by has_lol_emoji",
    y = "has_lol_emoji",
    x = "Not_yet_active_prev percentage",
    fill = "has_lol_emoji"
  )

# Boxplot of Not_yet_active percentage by has_shine_emoji
ggplot(df, aes(
  y = factor(has_shine_emoji),
  x = share_not_yet_active_pre,
  fill = factor(has_shine_emoji)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by has_shine_emoji",
    y = "has_shine_emoji",
    x = "Not_yet_active_prev percentage",
    fill = "has_shine_emoji"
  )

###### Boxplot of posts time settings ######

# Boxplot of Not_yet_active percentage by hour_band
ggplot(df, aes(
  y = factor(hour_band, labels = c("afternoon", "evening", "morning")),
  x = share_not_yet_active_pre,
  fill = factor(hour_band, labels = c("afternoon", "evening", "morning"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by hour_band",
    y = "Hour_band",
    x = "Not_yet_active_prev percentage",
    fill = "hour band"
  )



# Boxplot Not_yet_active_prev % by season and year
ggplot(df, aes(
  x = season,
  y = share_not_yet_active_pre,
  fill = season
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ season_year, nrow = 1) +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by season and year",
    x = "Season",
    y = "Not_yet_active_prev percentage",
    fill = "Season"
  )

ggplot(df, aes(
  x = season,
  y = share_not_yet_active_pre,
  fill = season
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_grid(section ~ season_year) +
  theme_minimal() +
  labs(
    title = "Not_yet_active_prev % by season, year and section",
    x = "Season",
    y = "Not_yet_active_prev percentage",
    fill = "Season"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

# Boxplot of Not_yet_active percentage by year and section
bw1 <- bwplot(share_not_yet_active_pre~ year_f | section, data = df)
xlims <- c("2023","2024","2025", "2026")
update(bw1, xlim = xlims, pch = "|")


# Boxplot of Not_yet_active percentage by year and section
bw2 <- bwplot(share_not_yet_active_pre~ as.factor(month) | as.factor(year), data = df)
xlims <- c("gen","feb","mar", "apr", "may","jun","jul", "aug", "sep","oct","nov", "dec")
update(bw2, xlim = xlims, pch = "|")


# Boxplot of Not_yet_active percentage by year and section
bw3 <- bwplot(share_not_yet_active_pre~ season | as.factor(year), data = df)
xlims <- c("spring", "summer", "autumn", "winter")
update(bw3, xlim = xlims, pch = "|")

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
  filter(n_total_activation > 0)

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
  filter(n_total_activation > 0)

mod_bb_complex <- glmmTMB(
  cbind(n_not_yet_active_pre, n_already_known_pre) ~
    year +
    season +
    audio_type +
    mentions_singer +
    mentions_alice +
    days_since_last_post +
    is_sanremo +
    hour_band +
    has_call_to_action +
    mentions_fashion +
    topic_Podcast + mentions_aimone + topic_YouTube+ is_ironic+ same_topic_last_10posts+ 
   avg_words_per_sentence+ location+ section+ mentions_creators+ 
video_duration_filled+ adv_density_last30d+ mentions_claudio+ mentions_famous +
topic_Theatre+ question_marks, 
  family = betabinomial(),
  data = train_df
)

summary(mod_bb_complex)

mod_bb_complex <- glmmTMB(
  cbind(n_not_yet_active_pre, n_already_known_pre) ~
    year +
    season +
    audio_type +
    mentions_singer +
    mentions_alice +
    days_since_last_post +
    is_sanremo +
    hour_band +
    has_call_to_action +
    mentions_fashion 
    , 
  family = betabinomial(),
  data = train_df
)

summary(mod_bb_complex)

# Remove topic_Podcast, mentions_aimone, topic_YouTube, is_ironic, same_topic_last_10posts, 
# avg_words_per_sentence, location, section, mentions_creators, 
# video_duration_filled, adv_density_last30d, mentions_claudio, mentions_famous
# topic_Theatre, question_marks

mod_no_alice <- update(
  mod_bb_complex,
  . ~ . - mentions_alice
)

anova(mod_no_alice, mod_bb_complex)
# Remove mentions alice
mod_bb_simple <- glmmTMB(
  cbind(n_not_yet_active_pre, n_already_known_pre) ~
    year +
    season +
    audio_type +
    mentions_singer +
    days_since_last_post +
    hour_band +
    has_call_to_action +
    mentions_fashion 
  , 
  family = betabinomial(),
  data = train_df
)

summary(mod_bb_simple)

mod_no_days_posts <- update(
  mod_bb_simple,
  . ~ . - days_since_last_post
)

anova(mod_no_days_posts, mod_bb_simple)
# Remove days_since_last posts

mod_full <- glmmTMB(
  cbind(n_not_yet_active_pre, n_already_known_pre) ~
    year +
    season +
    audio_type +
    mentions_singer +
    has_call_to_action +
    mentions_fashion, 
  family = betabinomial(),
  data = train_df
)

summary(mod_full)
# Remove hour_band and test mentions_fashion

mod_no_fashion <- update(
  mod_full,
  . ~ . - mentions_fashion
)

anova(mod_no_fashion, mod_full)
# Keep mentions fashion

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


train_df$share_obs <- train_df$n_not_yet_active_pre / 
  (train_df$n_not_yet_active_pre + train_df$n_already_known_pre)

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

#### ODDS RATIO ####

coefs <- summary(mod_final)$coefficients$cond

or_table <- data.frame(
  term = rownames(coefs),
  estimate = coefs[, "Estimate"],
  OR = exp(coefs[, "Estimate"]),
  CI_low = exp(coefs[, "Estimate"] - 1.96 * coefs[, "Std. Error"]),
  CI_high = exp(coefs[, "Estimate"] + 1.96 * coefs[, "Std. Error"]),
  p_value = coefs[, "Pr(>|z|)"]
)

or_table

#### R^2 ####
library(dplyr)


# 1. Allinea i factor/logical del test al train


rhs_vars <- c(
  "year",
  "season",
  "audio_type",
  "mentions_singer",
  "has_call_to_action",
  "mentions_fashion"
)

fix_to_train_levels <- function(x, train_x) {
  
  train_levels <- levels(factor(train_x))
  x_chr <- trimws(as.character(x))
  x_low <- tolower(x_chr)
  
  if (all(train_levels %in% c("FALSE", "TRUE"))) {
    x_chr <- case_when(
      x_low %in% c("true", "t", "1", "yes", "y", "si", "sì") ~ "TRUE",
      x_low %in% c("false", "f", "0", "no", "n") ~ "FALSE",
      TRUE ~ NA_character_
    )
  }
  
  factor(x_chr, levels = train_levels)
}

for (v in rhs_vars) {
  train_df[[v]] <- factor(train_df[[v]])
  test_df[[v]] <- fix_to_train_levels(test_df[[v]], train_df[[v]])
}

sapply(test_df[rhs_vars], function(x) sum(is.na(x)))


# 2. Predizione manuale per mod_full


manual_predict_glmmTMB <- function(model, newdata) {
  
  beta <- fixef(model)$cond
  
  rhs_form <- ~ year +
    season +
    audio_type +
    mentions_singer +
    has_call_to_action +
    mentions_fashion
  
  X <- model.matrix(rhs_form, data = newdata)
  
  missing_cols <- setdiff(names(beta), colnames(X))
  
  if (length(missing_cols) > 0) {
    for (mc in missing_cols) {
      X <- cbind(X, setNames(data.frame(0), mc))
    }
  }
  
  X <- X[, names(beta), drop = FALSE]
  
  eta <- drop(X %*% beta)
  
  plogis(eta)
}

# 3. Prepara test set e calcola predizioni


test_eval <- test_df %>%
  mutate(
    n_total = n_not_yet_active_pre + n_already_known_pre,
    share_obs = n_not_yet_active_pre / n_total
  ) %>%
  filter(n_total > 0) %>%
  filter(if_all(all_of(rhs_vars), ~ !is.na(.)))

test_eval$pred_full <- manual_predict_glmmTMB(
  model = mod_full,
  newdata = test_eval
)

test_eval$pred_full_clip <- pmin(
  pmax(test_eval$pred_full, 1e-6),
  1 - 1e-6
)
metriche <- function(obs, pred, w) {
  
  obs_mean_w <- weighted.mean(obs, w = w, na.rm = TRUE)
  
  weighted_r2 <- 1 -
    sum(w * (obs - pred)^2, na.rm = TRUE) /
    sum(w * (obs - obs_mean_w)^2, na.rm = TRUE)
  
  data.frame(
    RMSE = sqrt(weighted.mean((obs - pred)^2, w = w, na.rm = TRUE)),
    MAE = weighted.mean(abs(obs - pred), w = w, na.rm = TRUE),
    Bias = weighted.mean(obs - pred, w = w, na.rm = TRUE),
    Weighted_R2 = weighted_r2
  )
}
perf_full <- metriche(
  obs = test_eval$share_obs,
  pred = test_eval$pred_full_clip,
  w = test_eval$n_total
)

perf_full


#### Calibration plot ####

library(dplyr)
library(ggplot2)


# 3. Prepara test set + predizioni manuali


test_calib_data <- test_df %>%
  mutate(
    n_total = n_not_yet_active_pre + n_already_known_pre,
    obs_share = n_not_yet_active_pre / n_total
  ) %>%
  filter(n_total > 0) %>%
  filter(if_all(all_of(rhs_vars), ~ !is.na(.)))

test_calib_data$pred_share <- manual_predict_glmmTMB(
  model = mod_full,
  newdata = test_calib_data
)

test_calib_data <- test_calib_data %>%
  mutate(
    pred_share = pmin(pmax(pred_share, 1e-6), 1 - 1e-6)
  )


# 4. Calibration table


test_calib <- test_calib_data %>%
  mutate(
    pred_bin = ntile(pred_share, 5)
  ) %>%
  group_by(pred_bin) %>%
  summarise(
    mean_pred = weighted.mean(pred_share, n_total, na.rm = TRUE),
    obs_rate = sum(n_not_yet_active_pre, na.rm = TRUE) /
      sum(n_total, na.rm = TRUE),
    n_posts = n(),
    n_users = sum(n_total, na.rm = TRUE),
    .groups = "drop"
  )

test_calib


# 5. Calibration plot


ggplot(test_calib, aes(x = mean_pred, y = obs_rate, size = n_users)) +
  geom_point() +
  geom_text(
    aes(label = paste0("n=", n_posts)),
    vjust = -1,
    size = 3
  ) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    x = "Mean predicted Not yet active share",
    y = "Observed Not yet active share",
    size = "Users",
    title = "Calibration plot for Not yet active activation model",
    subtitle = "Test set"
  ) +
  theme_minimal()
