import csv
import numpy as np
from operator import itemgetter

###imports list of season games
inputs_raw = []
with open('team_game_season_2014_2015.csv', 'r') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for row in reader:
        inputs_raw.append([entry for entry in row])
csvfile.close()


###imports team legend
team_legend = []
with open('Team_legend.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        team_legend.append([entry for entry in row])
csvfile.close()


team_index = []
for team in team_legend:
    team_index.append(team[0])

###replace team name by abbreviation, add home game indicator
for game in inputs_raw:
    game[0] = team_legend[team_index.index(game[0])][1]

    if game[1][12:14] == 'vs':
        game.append('home')
    else:
        game.append('away')
    game[1] = game[1][0:10]

team_index.clear()
for team in team_legend:
    team_index.append(team[1])

###sort games by date'''
inputs_raw.sort(key=itemgetter(1))


###replace dates by ordered numbers
date_list = []
for game in inputs_raw:
    date_list.append(game[1])

date_numbered = [1]*len(date_list)
date_numbered[0] = 0
counter = 0
for i in range(1, len(date_list)):
    if date_list[i] == date_list[i-1]:
        date_numbered[i] = counter
    else:
        counter += 1
        date_numbered[i] = counter

for i in range(0, len(inputs_raw)):
    inputs_raw[i][1] = date_numbered[i]

###reducing columns in inputs_raw
for game in inputs_raw:
    # game.pop(21) #FOW%
    # game.pop(18) #PK%
    # game.pop(15) #PP%
    game.pop(6)  #ties
    game.pop(3)  #games played

#inputs_raw structure:
    #0 - team 1
    #1 - date
    #2 - team 2
    #3 - wins
    #4 - loss
    #5 - overtime loss
    #6 - points
    #7 - goals for
    #8 - goals against
    #9 - shots for
    #10 - shots against
    #11 - power play goals
    #12 - pp opp
    #13 - PP%
    #14 - TS
    #15 - PPGA
    #16 - PK%
    #17 - FOW
    #18 - FOL
    #19 - FOW%
    #20 - home/away indicator

###create training_game subset from input games
training_games = []
for game in inputs_raw:
    if game[20] == 'home':
        training_games.append(game[0:4])


for game in inputs_raw:
    game.pop(20) #home/away indicator
    game.pop(2) #opposing team


###building matrix to aggregate
aggregate_team = [[0 for i in range(len(inputs_raw[0])-2)] for j in range(30)]
data = []
max_agg_factor = 1    # the starting weight given to the current example when averaging
min_agg_factor = 0.1  # the minimum weight given to the current example when averaging
agg_factor_decay = 0.9 # decay factor reducing the weight given to the current game

for game in training_games:
    data.append([])
    for row in inputs_raw:
        if game[1] == row[1]:

            agg_factor = max(max_agg_factor, min_agg_factor)
            max_agg_factor = max_agg_factor * agg_factor_decay

            if game[0] == row[0]:
                data[-1][0:0] = aggregate_team[team_index.index(row[0])]  # Note we enter the game before aggregating to not include the results of the game in the inputs
                for i in range(len(aggregate_team[0])):
                    aggregate_team[team_index.index(row[0])][i] = agg_factor * float(row[i+2]) + (1-agg_factor) * aggregate_team[team_index.index(row[0])][i]
                    #aggregate_team[team_index.index(row[0])][i] += int(row[i+2])
            if game[2] == row[0]:
                data[-1].extend(aggregate_team[team_index.index(row[0])])
                for i in range(len(aggregate_team[0])):
                    aggregate_team[team_index.index(row[0])][i] = agg_factor * float(row[i+2]) + (1-agg_factor) * aggregate_team[team_index.index(row[0])][i]
                    #aggregate_team[team_index.index(row[0])][i] += int(row[i+2])

    data[-1][0:0] = [int(team_legend[team_index.index(game[2])][i]) for i in range(2, len(team_legend[0]))]
    data[-1][0:0] = [int(team_legend[team_index.index(game[0])][i]) for i in range(2, len(team_legend[0]))]
    data[-1].insert(0, int(game[3]))


# for i in range(30):
#     print(data[i][61:], training_games[i])
#     if i== 20:
#         print("-----------")
# Exclude a few games at the beginning of the season to avoid games where the team stats are all 0
nb_skipped = 20
data = data[nb_skipped:]

#data structure:
    # 0 - win indicator
    # 1:30 - team home
    # 31:60 - team away
    # 61 - Wins (team home)
    # 62 - Losses
    # 63 - Overtime Losses
    # 64 - Points
    # 65 - Goals For
    # 66 - Goals Against
    # 67 - Shots For
    # 68 - Shots Againsts
    # 69 - PPG
    # 70 - PP opp
    # 71 - PP%
    # 72 - TS
    # 73 - PPGA
    # 74 - PK%
    # 75 - FOW
    # 76 - FOL
    # 77 - FOW%
    # 77 - Wins (team away)
    # 78 - Losses
    # 79 - Overtime Losses
    # 80 - Points
    # 81 - Goals For
    # 82 - Goals Against
    # 83 - Shots For
    # 84 - Shots Againsts
    # 85 - PPG
    # 86 - PP opp
    # 87 - PP%
    # 88 - TS
    # 89 - PPGA
    # 90 - PK%
    # 91 - FOW
    # 92 - FOL
    # 93 - FOW%


#adding percentage variables

teams_inputs = np.array(data)[:, 0:61]  # These should not be normalized (team numbers)
stats_inputs = np.array(data)[:, 61:]

# for i in range(4):
#     print(teams_inputs[i])
#     print(stats_inputs[i])

normalized_inputs = (stats_inputs - np.mean(stats_inputs, 0)) / np.std(stats_inputs, 0, ddof=1)

# print(np.mean(normalized_inputs, 0))
# print(np.std(normalized_inputs, 0, ddof=1))
#
# for i in range(4):
#     print(normalized_inputs[i])

final_inputs = np.concatenate((teams_inputs, normalized_inputs), 1)

# for i in range(4):
#     print(final_inputs[i])

np.savetxt(
    'InputData2014-15_Final.csv',           # file name
    final_inputs,                # array to save
    fmt='%.15f',             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    comments='# ',          # character to use for comments
  )

print('end')