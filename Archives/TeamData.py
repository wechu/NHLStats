__author__ = 'Jonah'

import urllib.request
import re
import csv

url = "http://www.nhl.com/stats/team?season=20142015&gameType=2&viewName=summary"
website = urllib.request.urlopen(url)
readwebsite = website.read()

pattern='<td colspan="1" style="text-align: left;" rowspan="1"><a style="display:block;min-width:100px;" onclick="loadTeamSpotlight\\(jQuery\\(this\\)\\);" rel=".*?" href="javascript:void\\(0\\);">(.*?)</a></td>'\
        '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: center;" class="active" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: center;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'\
        '.*?<td colspan="1" style="text-align: right;" rowspan="1">(.*?)</td>'

teamdata = (re.findall(pattern,str(readwebsite)))
print(teamdata)


file = open('TeamData2014-15.csv','w')
writer = csv.writer(file, lineterminator='\n')
writer.writerows(teamdata)
file.close()
