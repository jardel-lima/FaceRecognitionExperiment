import random

folds = {}
fold_extra = {}
fold_prefix = 'fold_'

subjects = {}
subject_prefix = 'subject'

#subjects' ids - [1,2,3,4,5,6,7,8,9,10]
subject_ids = [x for x in range(1, 16)]

pics = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy',
        'surprised', 'wink']

#randomicly rearange subjects's ids - eg. [2,3,6,9,10,1,5,6,]
for i in range(1, 16):
    subject_id = subject_prefix+'{0:0>2}'.format(str(i))
    subjects[subject_id] = random.sample(pics, 11)

#create folds - the firts 2 numbers of each subject will form the first fold, the next
#2 numbers will form the second fold, and so on.
for j in range(5):
    fold_aux = {}
    fold_extra_aux = {}
    for k, v in subjects.items():
        fold_aux[k] = v[2*j:2*(j+1)]

    fold_id = fold_prefix+str(j+1)
    folds[fold_id] = fold_aux

for k, v in subjects.items():
    fold_extra[k] = v[len(v) - 1]

with open('five_fold_files.txt','w') as f:

    for fold in folds.values():
        for k, ids in fold.items():
            f.write(ids[0]+','+ids[1]+'/')
        f.write('\n')
    for k, v in fold_extra.items():
        f.write( v + '/')
