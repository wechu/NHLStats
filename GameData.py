__author__ = 'Jonah'
import urllib.request
import re
import csv

# requsting the website

datalist = []
for i in range(1, 42):
    url = "http://www.nhl.com/stats/game?fetchKey=20152ALLSATAll&viewName=summary&sort=gameDate&gp=1&pg={0}".format(i)
    website = urllib.request.urlopen(url)
    readwebsite = website.read()

    pattern='<a style="display:block;min-width:60px;" href="http://www.nhl.com/scores/htmlreports/20142015/.*?.HTM">(.*?)</a></td>'\
            '.*?<td colspan="1" style="text-align: left;" rowspan="1"><a style="display:block;min-width:100px;" onclick="loadTeamSpotlight\\(jQuery\\(this\\)\\);" rel=".*?" href="javascript:void\\(0\\);">(.*?)</a></td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<a style="display:block;min-width:100px;" onclick="loadTeamSpotlight\\(jQuery\\(this\\)\\);" rel=".*?" href="javascript:void\\(0\\);">(.*?)</a>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: left;" rowspan="1">(.*?)</td>'\
            '.*?<a style="display:block;min-width:100px;" href="/ice/player.htm\\?id=.*?">(.*?)</a>'\
            '.*?<a style="display:block;min-width:100px;" href="/ice/player.htm\\?id=.*?">(.*?)</a>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
            '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'

    pattern = re.compile(pattern)
    datalist.append(re.findall(pattern,str(readwebsite)))

for x in datalist:
     print(x)
     print('-------------')

file = open('GameData2014-15.csv','w')
writer = csv.writer(file, lineterminator='\n')

for page in datalist:
    writer.writerows(page)
file.close()