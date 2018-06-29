import requests

mid = 'm.0h37k4m'
result = requests.get(f'http://140.109.19.67:7705/kb_query?mid={mid}')
result = result.text.replace('<pre>','').replace('</pre>','')
#print(result)
lines = result.split('\n')
for line in lines:
    tokens = line.split('\t')
    print(tokens)
    break
#print(result)

