# use in staging server to create a small web dashboard with bloom filters stats

from aioserver import Application
app = Application()

@app.get('/')
async def index(request):
    reply = ""
    with open('/home/archiveteam/dashboard.txt', 'rt') as file:
        reply = file.read()
    return 200, {'Content-Type': 'text/html; charset=utf-8'}, "<html><body><style>body{font-family: monospace}</style>" + reply + "</body></html>"
        
app.run(host='0.0.0.0', port=80)
