

fn = 'model.txt'
f = open(fn, 'r')

model = f.readlines()

for ln in model:
    dat = ln.split(',')
    print dat
