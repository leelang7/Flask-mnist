import requests

url = 'http://127.0.0.1:8080'

img = 'img/img_two.png'

params = {'img' : img}
res = requests.get(url, params=params)
if res.status_code == 200:
    result = res.json()
    print(result)
else:
    print('failed')
