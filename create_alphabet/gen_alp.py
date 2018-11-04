import glob

path = './*.txt' #读入txt的路径
PathList = glob.glob(path)

alp_set=set()
print(PathList)
for file in PathList:
    with open(file, 'r') as f:
        first_line = f.readline()
        for c in first_line:
            alp_set.add(c)
    f.close()
for i in alp_set:
    print(i, end='')
alp_res='game_key.py'
with open(alp_res,'w') as f:
    for i in alp_set:
        print(i, end='',file=f)
f.close()