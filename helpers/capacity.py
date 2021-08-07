# use in staging server to return disk capacity level
import shutil

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
    stat = '{"' + stat.replace("=",'":').replace(", ",', "') + "}"
    return 200, {'Content-Type': 'application/json; charset=utf-8'},  stat
        
app.run(host='0.0.0.0', port=8080)