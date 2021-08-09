# use in staging server to return disk capacity level
import shutil
import json
import os

from aioserver import Application
app = Application()

# Path
path = "/home/archiveteam/CAH/gpujobs"
# Get the disk usage statistics
# about the given path
@app.get('/disk')
async def index(request):
    stat = str(shutil.disk_usage(path))
    stat = stat.split("(")[1].split(")")[0]
    stat = '{"' + stat.replace('=','":').replace(', ',', "') + '}'
    json_object = json.loads(stat)
    json_object["utilization"] =  round(json_object["used"]/json_object["total"], 2)
    json_object["jobscount"] = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

    return 200, {'Content-Type': 'application/json; charset=utf-8'},  json.dumps(json_object)
        
app.run(host='0.0.0.0', port=8080)