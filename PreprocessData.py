import csv
import numpy as np
from operator import itemgetter

class PreProcessing:
    def __init__(self, year):

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

        ###replace team name by abbreviation, add home game indicator
        for game in self.inputs_raw:
            game[0] = self.team_legend[self.team_index.index(game[0])][1]

            if game[1][12:14] == 'vs':
                game.insert(22,'home')
            else:
                game.insert(22,'away')
            game[1] = game[1][0:10]

        self.team_index.clear()
        for team in self.team_legend:
            self.team_index.append(team[1])

        ###sort games by date'''
        self.inputs_raw.sort(key=itemgetter(1))


        ###replace dates by ordered numbers
        date_list = []
        for game in self.inputs_raw:
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

        for i in range(0, len(self.inputs_raw)):
            self.inputs_raw[i][1] = date_numbered[i]

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

        # for game in self.inputs_raw:
        #     game.pop(20) #home/away indicator
        # #     game.pop(2) #opposing team


        ## create list of differences
        self.inputs_diff = []
        for game in self.inputs_raw:
            self.inputs_diff.append([])
            self.inputs_diff[-1].extend(game[0:2])
            self.inputs_diff[-1].append(game[3])
            self.inputs_diff[-1].append(float(game[6])/2)
            self.inputs_diff[-1].append(float(game[7])-float(game[8]))
            self.inputs_diff[-1].append(float(game[9])-float(game[10]))
            self.inputs_diff[-1].append(float(game[11])-float(game[15]))
            self.inputs_diff[-1].append(float(game[12])-float(game[14]))
            self.inputs_diff[-1].append(float(game[17])-float(game[18]))
            self.inputs_diff[-1].extend(game[23:28])
            self.inputs_diff[-1].extend(game[30:35])

        print(self.inputs_diff[0])

    def aggregation(self, game_set):
        ###building matrix to aggregate
        aggregate_team = [[0 for i in range(len(self.inputs_raw[0])-2)] for j in range(len(self.team_index))]
        data = []
        max_agg_factor = 1    # the starting weight given to the current example when averaging
        min_agg_factor = 0.1  # the minimum weight given to the current example when averaging
        agg_factor_decay = 0.9 # decay factor reducing the weight given to the current game


        for game in game_set:
            data.append([])
            for row in self.inputs_raw:
                if game[1] == row[1]:

                    agg_factor = max(max_agg_factor, min_agg_factor)
                    max_agg_factor = max_agg_factor * agg_factor_decay

                    if game[0] == row[0]:
                        data[-1][0:0] = aggregate_team[self.team_index.index(row[0])]  # Note we enter the game before aggregating to not include the results of the game in the inputs
                        for i in range(len(aggregate_team[0])):
                            aggregate_team[self.team_index.index(row[0])][i] = agg_factor * float(row[i+2]) + (1-agg_factor) * aggregate_team[self.team_index.index(row[0])][i]
                            #aggregate_team[self.team_index.index(row[0])][i] += int(row[i+2])
                    if game[2] == row[0]:
                        data[-1].extend(aggregate_team[self.team_index.index(row[0])])
                        for i in range(len(aggregate_team[0])):
                            aggregate_team[self.team_index.index(row[0])][i] = agg_factor * float(row[i+2]) + (1-agg_factor) * aggregate_team[self.team_index.index(row[0])][i]
                            #aggregate_team[self.team_index.index(row[0])][i] += int(row[i+2])

            data[-1][0:0] = [int(self.team_legend[self.team_index.index(game[2])][i]) for i in range(2, len(self.team_legend[0]))]
            data[-1][0:0] = [int(self.team_legend[self.team_index.index(game[0])][i]) for i in range(2, len(self.team_legend[0]))]
            data[-1].insert(0, int(game[3]))

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
        return data

    def valid_builder(self, step):
        if step == 0:
            return self.aggregation(self.training_games), None

        #creates trainings sets and testings sets for cross-validation
        training_sets = [[self.training_games[i] for i in range(len(self.training_games)) if i % step != j] for j in range(step)]
        data = self.aggregation(self.training_games)

        aggregate_training_sets = [self.aggregation(lst) for lst in training_sets]
        aggregate_testing_sets = [data[i::step] for i in range(step)]

        return aggregate_training_sets, aggregate_testing_sets

    def normalize(self, data, data_test):

        teams_inputs = np.array(data)[:, 0:61]  # These should not be normalized (team numbers)
        stats_inputs = np.array(data)[:, 61:]

        mean = np.mean(stats_inputs, 0)
        std = np.std(stats_inputs, 0, ddof=1)

        normalized_inputs = (stats_inputs - mean) / std
        final_inputs = np.concatenate((teams_inputs, normalized_inputs), 1)


        if data_test != None:
            teams_inputs_test = np.array(data_test)[:, 0:61]  # These should not be normalized (team numbers)
            stats_inputs_test = np.array(data_test)[:, 61:]
            normalized_inputs_test = (stats_inputs_test - mean) / std
            final_inputs_test = np.concatenate((teams_inputs_test, normalized_inputs_test), 1)

        else:
            final_inputs_test = []

        return final_inputs, final_inputs_test

    def export_data(self, file_name, save_data):
        np.savetxt(
        str(file_name) + '.csv',           # file name
        save_data,                # array to save
        fmt='%.15f',             # formatting, 2 digits in this case
        delimiter=',',          # column delimiter
        newline='\n',           # new line character
        comments='# ',          # character to use for comments
        )

def preprocessing_cross_valid(year, nb_folds):
    # Creates cross validation sets
    p = PreProcessing(year)
    data, data_test = p.valid_builder(nb_folds)
    normalized_data = []
    normalized_test_data = []

    for i in range(nb_folds):
        normalized_data_a, normalized_test_data_a = p.normalize(data[i], data_test[i])
        normalized_data.append(normalized_data_a)
        normalized_test_data.append(normalized_test_data_a)

    return normalized_data, normalized_test_data


def preprocessing_final(year, file_name):
    # Preprocesses all data
    # Eg: preprocessing_final(2013, 'test')
    p = PreProcessing(year)
    data, data_test = p.valid_builder(0)
    normalized_data, normalized_test_data = p.normalize(data, data_test)
    p.export_data(file_name, normalized_data)
    print('Preprocessing for year ' + str(year) + '-' + str(year+1) + ' completed')

p = PreProcessing(2014)