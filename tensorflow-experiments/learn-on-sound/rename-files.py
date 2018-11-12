import os
path = 'sound/tr/kick'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, '1-'+str(i)+'.wav'))
    i = i+1