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
    stat = shutil.disk_usage(path)
    return 200, {'Content-Type': 'text/html; charset=utf-8'}, "<html><body><style>body{font-family: monospace}</style>" + stat + "</body></html>"
        
app.run(host='0.0.0.0', port=80)