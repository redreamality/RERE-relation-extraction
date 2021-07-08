import json

folders = ['NYT10-HRL','NYT11-HRL','ske2019']
folders=['ske2019']
ds = ['train','dev','test']
fnratio = [0]
for r in fnratio:
    for folder in folders:
        for dd in ds:
            rel_set = set()
            data = []
            name= f'FNexp/{folder}@{r}/new_{dd}.json'
            print(name)
            load_dic=[]
            with open('./'+name,'r',encoding='utf-8') as load_f:
                for l in load_f.readlines():
                    a = json.loads(l)
                    print(l)
                    if not a['relationMentions']:
                        continue
                    line = {
                            'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                            'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
                        }
                    if not line['triple_list']:
                        continue
                    data.append(line)
                    for rm in a['relationMentions']:
                        if rm['label'] != 'None':
                            rel_set.add(rm['label'])
            with open(f'FNexp/{folder}@{r}/{dd}_triples.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        id2predicate = {i:j for i,j in enumerate(sorted(rel_set))}
        predicate2id = {j:i for i,j in id2predicate.items()}


        with open(f'FNexp/{folder}@{r}/rel2id.json', 'w', encoding='utf-8') as f:
            json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)

