import csv
import numpy as np
import math
from operator import itemgetter

# Change self.team_index

# Current data legend
# 1 #Wins
# 2 #Points
# 3 #Goals for - Goals against
# 4 #Shots for - Shots against
# 5 #Power play goals - power play goals against
# 6 #Power play opportunities - times shorthanded
# 7 #Face off wins - face off losts
# 8-12 #SATs (corsi)
# 13-17 #USATs (fenwick)

class PreProcessing:
    def __init__(self, year):

        self.year = year

        ###imports team legend
        self.team_legend = []
        with open('Team_legend.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.team_legend.append([entry for entry in row])
        csvfile.close()

        self.team_index = []
        for team in self.team_legend:
            self.team_index.append(team[0])

        ###imports list of season games
        self.inputs_raw = []
        with open('team_game_season_' + str(year) + '_' + str(year+1) + '.csv', 'r') as csvfile:
            next(csvfile)
            reader = csv.reader(csvfile)
            for row in reader:
                self.inputs_raw.append([entry for entry in row])
        csvfile.close()

        ###imports advanced stats of season games
        self.advanced_stats = []
        with open('advanced_team_stats_' + str(year) + '_' + str(year+1) + '.csv', 'r') as csvfile:
            next(csvfile)
            reader = csv.reader(csvfile)
            for row in reader:
                self.advanced_stats.append([entry for entry in row])
        csvfile.close()

        for game in self.inputs_raw:
            for stat in self.advanced_stats:
                if game[0] == stat[0] and game[1][0:10] == stat[1][0:10] and game[2] == stat[2]:
                    game.extend(stat[3:])


        max_length = max(len(game) for game in self.inputs_raw)
        for game in self.inputs_raw:
            if len(game) < max_length:
                game.extend([0]*(max_length-len(game)))
                print(game)


        ###replace team name by abbreviation, add home game indicator
        for game in self.inputs_raw:
            game[0] = self.team_legend[self.team_index.index(game[0])][1]

            if game[1][12:14] == 'vs':
                game.insert(22, 'home')
            else:
                game.insert(22, 'away')
            game[1] = game[1][0:10]

        self.team_index.clear()
        for team in self.team_legend:
            self.team_index.append(team[1])

        ###sort games by date'''
        self.inputs_raw.sort(key=itemgetter(1))

        ### keep only dates
        for game in self.inputs_raw:
            game[1] = game[1][0:10]

        ###reducing columns in self.inputs_raw
        for game in self.inputs_raw:
            game.pop(6)  #ties
            game.pop(3)  #games played

        #self.inputs_raw structure:
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
            #21 - SAT For
            #22 - SAT Agst
            #23 - SAT
            #24 - SAT Tied
            #25 - SAT Ahead
            #26 - SAT Behind
            #27 - SAT Close
            #28 - USAT For
            #29 - USAT Agst
            #30 - USAT
            #31 - USAT Tied
            #32 - USAT Ahead
            #33 - USAT Behind
            #34 - USAT Close


        ###create training_game subset from input games
        self.training_games = []
        for game in self.inputs_raw:
            if game[20] == 'home':
                self.training_games.append(game[0:4])

        ## create list of differences
        self.inputs_diff = []
        for game in self.inputs_raw:
            self.inputs_diff.append([])
            self.inputs_diff[-1].extend(game[0:2]) #Team 1, date,
            self.inputs_diff[-1].append(float(game[3])-0.5) #Wins
            self.inputs_diff[-1].append(float(game[6])-1) #Points
            self.inputs_diff[-1].append(float(game[7])-float(game[8])) #Goals for - Goals against
            self.inputs_diff[-1].append(float(game[9])-float(game[10])) #Shots for - Shots against
            self.inputs_diff[-1].append(float(game[11])-float(game[15])) #Power play goals - power play goals against
            self.inputs_diff[-1].append(float(game[12])-float(game[14])) #Power play opportunities - times shorthanded
            self.inputs_diff[-1].append(float(game[17])-float(game[18])) #Face off wins - face off losts
            self.inputs_diff[-1].extend(game[23:28]) #SATs (corsi)
            self.inputs_diff[-1].extend(game[30:35]) #USATs (fenwick)

    def delta_elo(self, k_factor, elo_1, elo_2, score_diff, margin_victory=True):
        #calculates the change in elo for team with elo_1
        expected_result = 1/(1+10**((elo_2-elo_1)/400))

        if score_diff > 0:
            actual_result = 1
            M = 2.2/(0.001*(elo_1-elo_2) +2.2)
        elif score_diff == 0:
            actual_result = 0.5
            M = 1
        else:
            actual_result = 0
            M = 2.2/(0.001*(elo_2-elo_1) +2.2)

        delta_elo = 0

        if not margin_victory:
            delta_elo += k_factor*(actual_result-expected_result)

        else:
            mov = math.log(abs(score_diff)+1)
            delta_elo += k_factor*mov*M*(actual_result-expected_result)

        return delta_elo

    def aggregation(self, game_set, first_year=False):

        if first_year:
            #1500 initial elo
            self.elo_team = [[1500 for i in range(len(self.inputs_diff[0])-2)] for j in range(len(self.team_index))]

        else:
            self.elo_team = []

            with open('elo_' + str(self.year) + '.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    self.elo_team.append([float(entry) for entry in row])
            csvfile.close()

        if len(self.elo_team) !=31:
            print('warning: missing a team in elo matrix')

        data = []
        k_factor = 10 #judgement call based on USCF approximation and Nate Silver's previous work with elo

        for game in game_set:
            data.append([])
            game_stats = []
            for row in self.inputs_diff:
                if game[1] == row[1]: #match time

                    if game[0] == row[0]: #if home team
                        data[-1][0:0] = self.elo_team[self.team_index.index(game[0])]  # Note: we enter the game before aggregating to not include the results of the game in the inputs
                        game_stats[0:0] = row[2:]

                    if game[2] == row[0]: #if away team
                        data[-1].extend(self.elo_team[self.team_index.index(game[2])])
                        game_stats.extend(row[2:])


            #updating elo
            for i in range(len(self.elo_team[0])):
                old_elo_1 = float(data[-1][i])
                old_elo_2 = float(data[-1][i+len(self.elo_team[0])])

                if i == 0 or i == 1:
                    delta = self.delta_elo(k_factor, old_elo_1, old_elo_2, float(game_stats[i]), False)
                    self.elo_team[self.team_index.index(game[0])][i] += delta
                    self.elo_team[self.team_index.index(game[2])][i] -= delta

                else:
                    delta = self.delta_elo(k_factor, old_elo_1, old_elo_2, float(game_stats[i]))
                    self.elo_team[self.team_index.index(game[0])][i] += delta
                    self.elo_team[self.team_index.index(game[2])][i] -= delta

            #testing elo changes
            # if game[0] == 'CBJ' or game[2] == 'CBJ':
            #     print(game)
            #     print(self.elo_team[self.team_index.index('CBJ')][0])
            #     pass

            # data[-1][0:0] = [int(self.team_legend[self.team_index.index(game[2])][i]) for i in range(2, len(self.team_legend[0]))]
            # data[-1][0:0] = [int(self.team_legend[self.team_index.index(game[0])][i]) for i in range(2, len(self.team_legend[0]))]
            # data[-1][0:0] = [self.team_index.index(game[2])]
            # data[-1][0:0] = [self.team_index.index(game[0])]
            data[-1].insert(0, int(game[3]))

        nb_skipped = 0 # is it still necessary to skip games?
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
        return data

    def export_elo(self, year, soft_reset=True):
        if soft_reset:
            elo = np.array(self.elo_team) * 2/3 + 1/3 * 1500

        else:
            elo = np.array(self.elo_team)


        np.savetxt(
        'elo_' + str(year+1) + '.csv',           # file name
        elo,                # array to save
        fmt='%.15f',             # formatting, 2 digits in this case
        delimiter=',',          # column delimiter
        newline='\n',           # new line character
        comments='# ',          # character to use for comments
        )

    def valid_builder(self, step, first_year=False):
        # Creates cross-validation sets
        # Returns list of all training sets and list of all testing sets
        # Note: if step is 0, then it is a single list (not list of lists)
        if step == 0:
            return self.aggregation(self.training_games, first_year), None

        #creates trainings sets and testings sets for cross-validation
        training_sets = [[self.training_games[i] for i in range(len(self.training_games)) if i % step != j] for j in range(step)]
        data = self.aggregation(self.training_games, first_year)

        aggregate_training_sets = [self.aggregation(lst, first_year) for lst in training_sets]
        aggregate_testing_sets = [data[i::step] for i in range(step)]

        return aggregate_training_sets, aggregate_testing_sets

def normalize(data, data_test):

    targets = np.array(data)[:, 0:1]  # These should not be normalized (targets)
    stats_inputs = np.array(data)[:, 1:]

    mean = np.mean(stats_inputs, 0)
    std = np.std(stats_inputs, 0, ddof=1)
    normalized_inputs = (stats_inputs - mean) / std
    final_inputs = np.concatenate((targets, normalized_inputs), 1)


    if data_test:
        targets_test = np.array(data_test)[:, 0:1]  # These should not be normalized (targets)
        stats_inputs_test = np.array(data_test)[:, 1:]
        normalized_inputs_test = (stats_inputs_test - mean) / std
        final_inputs_test = np.concatenate((targets_test, normalized_inputs_test), 1)

    else:
        final_inputs_test = []

    return final_inputs, final_inputs_test

def export_data(file_name, save_data):
    np.savetxt(
    str(file_name) + '.csv',           # file name
    save_data,                # array to save
    fmt='%.15f',             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    comments='# ',          # character to use for comments
    )

def preprocessing_cross_valid(year_start, year_end, nb_folds):
    # Creates cross validation sets over multiple years
    data_final = [[] for i in range(nb_folds)]
    data_test_final = [[] for i in range(nb_folds)]
    normalized_data = []
    normalized_test_data = []

    for j in range(year_start, year_end+1):
        p = PreProcessing(j)
        data, data_test = p.valid_builder(nb_folds, j == year_start)
        for k in range(nb_folds):
            data_final[k].extend(data[k])
            data_test_final[k].extend(data_test[k])

    for i in range(nb_folds):
        normalized_data_temp, normalized_test_data_temp = normalize(data_final[i], data_test_final[i])
        normalized_data.append(normalized_data_temp)
        normalized_test_data.append(normalized_test_data_temp)

    return normalized_data, normalized_test_data


def preprocessing_final(year_start, year_end, file_name=None, export=True):
    # Preprocesses all data
    # Eg: preprocessing_final(2013, 'test')
    data_final = []
    for i in range(year_start, year_end+1):
        p = PreProcessing(i)

        data, data_test = p.valid_builder(0, i == year_start)

        p.export_elo(i)
        data_final.extend(data)
        print('Preprocessing for year ' + str(i) + '-' + str(i+1) + ' completed')

    normalized_data = normalize(data_final, [])
    if export:
        export_data(file_name, normalized_data)

    return normalized_data

if __name__ == "__main__":
    pass
    preprocessing_final(2012, 2014, 't4')

    # preprocessing_cross_valid(2014, 2014, 10)
