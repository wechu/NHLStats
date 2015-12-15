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
date_numbered[0] = 1
counter = 1
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


def aggregate_team_specs(date_number):
    holder_list = []
    for game in inputs_raw:
        if game[0] == 'MTL' and game[1] < date_number:
             game.pop(18)
             game.pop(2)
             game.pop(1)
             game.pop(0)
             holder_list.append(game)

    print(holder_list)




    return

aggregate_team_specs(3)

#print(team_date_number_abv)




# for i in range(4):
#   print(inputs_raw[i])
#
# teams_inputs = np.array(inputs)[:, 0:2]  # These should not be normalized (team numbers)
# stats_inputs = np.array(inputs)[:, 2:]
#
# # for i in range(4):
# #     print(teams_inputs[i])
# #     print(stats_inputs[i])
#
# normalized_inputs = (stats_inputs - np.mean(stats_inputs, 0)) / np.std(stats_inputs, 0, ddof=1)
#
# # print(np.mean(normalized_inputs, 0))
# # print(np.std(normalized_inputs, 0, ddof=1))
# # for i in range(4):
# #     print(normalized_inputs[i])
#
# final_inputs = np.concatenate((teams_inputs, normalized_inputs), 1)
#
# # for i in range(4):
# #     print(final_inputs[i])
#
# file = open('InputData2014-15_Final.csv','w')
# writer = csv.writer(file, lineterminator='\n')
# for row in final_inputs:
#     writer.writerow(["{:.15f}".format(x) for x in row])  # Need to preserve as many decimals as possible
# file.close()
