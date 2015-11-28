import csv
import numpy as np
from operator import itemgetter

inputs_raw = []
with open('team_game_season_2014_2015.csv', 'r') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for row in reader:
        inputs_raw.append([(entry) for entry in row])
csvfile.close()

team_legend=[]
with open('Team_legend.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        team_legend.append([entry for entry in row])
csvfile.close()


team_index=[]
for team in team_legend:
    team_index.append(team[0])

for game in inputs_raw:
    game[0]=team_legend[team_index.index(game[0])][1]

    if game[1][12:14]=='vs':
        game.insert(24,1)
    else:
        game.insert(24,0)
    game[1]=game[1][0:10]

inputs_raw.sort(key=itemgetter(1))

del team_index[:]
for team in team_legend:
    team_index.append(team[1])

print(inputs_raw)




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
