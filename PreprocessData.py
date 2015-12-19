import csv
import numpy as np
from operator import itemgetter

'''imports list of season games'''
inputs_raw = []
with open('team_game_season_2014_2015.csv', 'r') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for row in reader:
        inputs_raw.append([entry for entry in row])
csvfile.close()


'''imports team legend'''
team_legend = []
with open('Team_legend.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        team_legend.append([entry for entry in row])
csvfile.close()


team_index = []
for team in team_legend:
    team_index.append(team[0])

'''replace team name by abbreviation, add home game indicator'''
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

'''sort games by date'''
inputs_raw.sort(key=itemgetter(1))


'''replace dates by ordered numbers'''
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

'''reducing columns in inputs_raw'''
for game in inputs_raw:
    game.pop(21) #FOW%
    game.pop(18) #PK%
    game.pop(15) #PP%
    game.pop(6)  #ties


'''create training_game subset from input games'''
training_games = []
for game in inputs_raw:
    if game[18] == 'home':
        training_games.append(game[0:5])
for game in training_games:
    game.pop(3) #games played


for game in inputs_raw:
    game.pop(18) #home/away indicator
    game.pop(2) #opposing team


''' building matrix to aggregate '''
aggregate_team = [[0 for i in range(15)] for j in range(30)]
data = []

for game in training_games:
    data.append([])
    for row in inputs_raw:
        if game[1] == row[1]:
            if game[0] == row[0]:
                for i in range(len(aggregate_team[0])):
                    aggregate_team[team_index.index(row[0])][i] += int(row[i+2])
                data[-1][0:0] = aggregate_team[team_index.index(row[0])]

            if game[2] == row[0]:
                for i in range(len(aggregate_team[0])):
                    aggregate_team[team_index.index(row[0])][i] += int(row[i+2])

                data[-1].extend(aggregate_team[team_index.index(row[0])])
    data[-1][0:0] = [int(team_legend[team_index.index(game[2])][i]) for i in range(2, len(team_legend[0]))]
    data[-1][0:0] = [int(team_legend[team_index.index(game[0])][i]) for i in range(2, len(team_legend[0]))]
    data[-1].insert(0, int(game[3]))


#data structure:
    # 0 - win indicator
    # 1:30 - team 1
    # 31:60 - team 2
    # 61 - GP (team1)
    # 62 - Wins
    # 63 - Losses
    # 64 - Overtime Losses
    # 65 - Points
    # 66 - Goals For
    # 67 - Goals Against
    # 68 - Shots For
    # 69 - Shots Againsts
    # 70 - PPG
    # 71 - PP opp
    # 72 - TS
    # 73 - PPGA
    # 74 - FOW
    # 75 - FOL
    # 76 - GP (team2)
    # 77 - Wins
    # 78 - Losses
    # 79 - Overtime Losses
    # 80 - Points
    # 81 - Goals For
    # 82 - Goals Against
    # 83 - Shots For
    # 84 - Shots Againsts
    # 85 - PPG
    # 86 - PP opp
    # 87 - TS
    # 88 - PPGA
    # 89 - FOW
    # 90 - FOL
    # 91 - PP% (team1)
    # 92 - PK% (team1)
    # 93 - FOW% (team1)
    # 94 - PP% (team2)
    # 95 - PK% (team2)
    # 96 - FOW% (team2)



#adding percentage variables
for agg_game in data:

    #home
    agg_game.append(agg_game[70]/agg_game[71])
    agg_game.append((agg_game[72]-agg_game[73])/agg_game[72])
    agg_game.append(agg_game[74]/(agg_game[74]+agg_game[75]))

    #away
    agg_game.append(agg_game[85]/agg_game[86])
    agg_game.append((agg_game[87]-agg_game[88])/agg_game[87])
    agg_game.append(agg_game[89]/(agg_game[89]+agg_game[90]))

print(data)

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

file = open('InputData2014-15_Final.csv', 'w')
writer = csv.writer(file, lineterminator='\n')
for row in final_inputs:
    writer.writerow(["{:.15f}".format(x) for x in row])  # Need to preserve as many decimals as possible
file.close()
