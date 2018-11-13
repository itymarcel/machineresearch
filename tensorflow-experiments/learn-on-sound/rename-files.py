import os
path = 'sound/claps_TRAINING'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, '4-'+str(i)+'.wav'))
    i = i+1