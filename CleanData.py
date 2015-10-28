import csv

# with open('TeamData2014-15.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         print(row)


extract_game_data = []
with open('GameData2014-15.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        #print(row)
        extract_game_data.append([row[i] for i in range(1,5)])

csvfile.close()

for i in range(len(extract_game_data)):
    extract_game_data[i].append(len(extract_game_data) - i)

teams = []
scores = []

for row in extract_game_data:
    teams.append([row[i] for i in (0,2,4)])
    scores.append([int(row[i]) for i in (1,3)])


legend_teams = ["ANAHEIM", "ARIZONA", "BOSTON", "BUFFALO", "CALGARY", "CAROLINA", "CHICAGO",
 "COLORADO", "COLUMBUS", "DALLAS", "DETROIT", "EDMONTON", "FLORIDA", "LOS ANGELES",
 "MINNESOTA", "MONTREAL", "NASHVILLE", "NEW JERSEY", "NY ISLANDERS", "NY RANGERS",
 "OTTAWA", "PHILADELPHIA", "PITTSBURGH", "SAN JOSE", "ST LOUIS", "TAMPA BAY",
 "TORONTO", "VANCOUVER", "WASHINGTON", "WINNIPEG"]

for row in teams:
    row[0] = legend_teams.index(row[0])
    row[1] = legend_teams.index(row[1])

print(teams)
print(scores)

file = open('TeamData2014-15_Clean.csv','w')
writer = csv.writer(file, lineterminator='\n')
writer.writerows(teams)
file.close()

file = open('GameData2014-15_Clean.csv','w')
writer = csv.writer(file, lineterminator='\n')
writer.writerows(scores)
file.close()