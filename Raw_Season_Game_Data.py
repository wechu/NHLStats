__author__ = 'Jonah'
# Scraping data from NHL statistics website

import urllib.request
import re
import csv


# variables
season_year=20142015
number_pages=50

datalist = []
for i in range(1, number_pages):
    url = "http://www.nhl.com/stats/team?reportType=game&report=teamsummary&season={0}&gameType=2&aggregate=0".format(season_year)
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

file = open('Raw_Season_Game_Data_{}.csv'.format(season_year),'w')
writer = csv.writer(file, lineterminator='\n')

for page in datalist:
    writer.writerows(page)
file.close()