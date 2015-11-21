import csv

XInput = []

with open('TeamData2014-15_Clean.csv') as csvT:
    reader = csv.reader(csvT)
    for row in reader:
        XInput.append(row)


Teaminfo = []
with open('TeamData2014-15.csv') as csvTe:
    reader = csv.reader(csvTe)
    for row in reader:
        Teaminfo.append(row)

#print(Teaminfo[0][1:])

for eachgame in XInput:
    eachgame.extend(Teaminfo[int(eachgame[0])][2:])
    eachgame.extend(Teaminfo[int(eachgame[1])][2:])


file = open('InputData2014-15_Raw.csv','w')
writer = csv.writer(file, lineterminator='\n')
writer.writerows(XInput)
file.close()
