import os


folders = ['NYT10-HRL','NYT11-HRL']
# fnratio = [ str(round(a*0.1, 1)) for a in range(1,6)]
for d in folders:
    # for radio in fnratio:
    cmd = f'python extraction.py {d}'
    print(cmd)
    with open('result.csv','a',encoding='utf8') as f: f.write('\n'+cmd+'\n')
    os.system(cmd)
for d in folders:
    # for radio in fnratio:
    cmd = f'python extraction.py {d}'
    print(cmd)
    with open('result.csv','a',encoding='utf8') as f: f.write('\n'+cmd+'\n')
    os.system(cmd)