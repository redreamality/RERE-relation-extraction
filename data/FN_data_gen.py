import utils,os,json,random
# folders = ['Wiki-KBP','WebNLG','NYT','NYT11-HRL','NYT10-HRL','ske2019'][3:]


folders = ['ske2019']
ds = ['train','valid','dev','test']
fnratio = [0.1,0.2,0.3,0.4,0.5]

rel_set = set()
for folder in folders:
    for r in fnratio:
        for d in ds:
            dname = f'{folder}'
            odname = f'FNexp/{folder}@{r}'
            print(os.path.isdir(odname))        
            if not os.path.isdir(odname): os.makedirs(odname)
            fn = f'{dname}/new_{d}.json'
            # print(fn)
            # print(os.path.exists(fn))
            if not os.path.exists(fn): continue
            li = utils.LoadJsons(fn)
            if d=='train':
                newli = []
                for dic in li:                    
                    newspos = []
                    if not dic['relationMentions']:continue
                    for spo in dic['relationMentions']:
                        if random.random()>float(r): 
                            newspos.append(spo)
                    dic['relationMentions'] = newspos
                    if not dic['relationMentions']:continue
                    newli.append(dic)
            else:
                newli = li
            utils.SaveList(map(lambda x:json.dumps(x, ensure_ascii=False), newli), f'{odname}/new_{d}.json')
            




