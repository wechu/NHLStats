import csv
import numpy as np

inputs = []
with open('InputData2014-15_Raw.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        inputs.append([float(entry) for entry in row])

csvfile.close()

# for i in range(4):
#    print(inputs[i])

teams_inputs = np.array(inputs)[:, 0:2]  # These should not be normalized (team numbers)
stats_inputs = np.array(inputs)[:, 2:]

# for i in range(4):
#     print(teams_inputs[i])
#     print(stats_inputs[i])

normalized_inputs = (stats_inputs - np.mean(stats_inputs, 0)) / np.std(stats_inputs, 0, ddof=1)

# print(np.mean(normalized_inputs, 0))
# print(np.std(normalized_inputs, 0, ddof=1))
# for i in range(4):
#     print(normalized_inputs[i])

final_inputs = np.concatenate((teams_inputs, normalized_inputs), 1)

# for i in range(4):
#     print(final_inputs[i])

file = open('InputData2014-15_Final.csv','w')
writer = csv.writer(file, lineterminator='\n')
for row in final_inputs:
    writer.writerow(["{:.15f}".format(x) for x in row])  # Need to preserve as many decimals as possible
file.close()