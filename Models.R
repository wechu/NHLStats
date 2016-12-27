
setwd("C:/Users/Wes/Documents/GitHub/NHLStats")


year = 2015
filename = paste0("team_game_season_", year, "_", year+1, ".csv")

### Read the raw season data
dat = read.csv(filename, fileEncoding="UTF-8-BOM")
# Sort by date
dat = dat[order(dat$Game), ]
# Remove duplicates (keep only vs., remove @)
dat = dat[grepl(".+vs.+", dat$Game), ]


### Get team legend
# team_legend maps team names to their 3-letter tags
team_legend = read.csv("team_legend.csv", fileEncoding="UTF-8-BOM", header = F)

rownames(team_legend) = team_legend$V1
team_legend = team_legend[2]


### Fix up data so that there is only Time, Home team, Away team and result
# Rename home teams with their 3-letter tag
dat$Home = team_legend[dat$Team, ]
# Keep only Time, Home, Away and Result columns
dat = dat[c("Home", "Opp.Team", "W")]
dat$Time = 1:nrow(dat)
colnames(dat) = c("Home", "Away", "Result", "Time")
# Reorder columns
dat = dat[c(4, 1, 2, 3)]

dat$Home = as.character(dat$Home)
dat$Away = as.character(dat$Away)


### Try rating systems
library(PlayerRatings)


ratings = glicko(dat, history=T)

ratings.value = ratings$history[,,1]
ratings.value = data.frame(t(ratings.value))


plot(ratings.value$ARI, type="l", ylim=c(1700, 2800))

for(i in 1:30) {
  lines(ratings.value[names(ratings.value)[i]], col="gray")
}
lines(ratings.value$MTL)



### Test effectiveness at prediction

# Train on all elements up to train_time then predict the next game
first_time = 600
results = data.frame(pred = rep(NA, nrow(dat)), target = logical(nrow(dat)))
results$target = (dat$Result == 1)
  
ratings= glicko(dat[dat$Time < first_time-2,])

# Do prediction on every game given the past, starting with first_time
for(train_time in first_time:nrow(dat)) {
  
  train_dat = dat[dat$Time == train_time-1, ]
  test_dat = dat[dat$Time == train_time, ]
  # Use status parameter to keep the old ratings and then add one more game
  ratings = glicko(train_dat, status=ratings$ratings, gamma=1, cval=30) 
  
  preds = predict(ratings, test_dat, tng = 10, gamma=1) # tng is min number of games each team has to have played
  
  results$pred[train_time] = preds
}

results = results[!is.na(results$pred), ]
results$pred_class = results$pred > 0.5

# Check out results
confusion = table(results$target, results$pred_class)
confusion
acc = (confusion[1] + confusion[3]) / sum(confusion)
acc

# Well the accuracy is worse than random guessing :/
# Need to tune a lot


###### ignore this
# test = glicko(dat[1:200, ])
# 
# pr = predict(test, dat[200:nrow(dat), ], tng=10)
# pr = pr > 0.5
# confusion = table(pr, dat$Result[200:nrow(dat)])
