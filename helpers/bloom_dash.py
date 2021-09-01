# use in staging server to create a small web dashboard with bloom filters stats

from aioserver import Application
app = Application()

@app.get('/')
async def index(request):
    reply = ""
    with open('/home/archiveteam/dashboard.txt', 'rt') as file:
        reply = file.read()
    return 200, {'Content-Type': 'text/html; charset=utf-8'}, "<html><body><style>body{font-family: monospace}</style>" + reply + "</body></html>"

@app.get('/stats')
async def index(request):
    uniques = []
    total = []
    clipped = []
    with open('/home/archiveteam/dashboard.txt', 'rt') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("M unique pairs"):
                chunks = line.split("<br/>")
                uniques.append(chunks[0].split(" ")[-1])
                total.append(chunks[1].split(" ")[-1])
                clipped.append(chunks[2].split(" ")[-1])

    reply = '{"total": {"uniques":' + uniques[0] + ',"pairs":' + total[0] + ',"clips":' + clipped[0] + '},'
    reply += '"day": {"uniques":' + uniques[1] + ',"pairs":' + total[1] + ',"clips":' + clipped[1] + '},'
    reply += '"week": {"uniques":' + uniques[2] + ',"pairs":' + total[2] + ',"clips":' + clipped[2] + "}}"

    return 200, {'Content-Type': 'application.json; charset=utf-8'}, reply

app.run(host='0.0.0.0', port=8080)
