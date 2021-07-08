import json
dic=dict()
dic['NYT10-HRL']={'thre_rc':0.4,'thre_ee':0.2}
dic['NYT11-HRL']={'thre_rc':0.56,'thre_ee':0.55}
dic['NYT21-HRL']={'thre_rc':0.56,'thre_ee':0.55}
dic['ske2019']={'thre_rc':0.5,'thre_ee':0.3}
# with open('config.json','w') as f:
print(dic)
with open('config.json','w') as f:
    json.dump(dic,f)