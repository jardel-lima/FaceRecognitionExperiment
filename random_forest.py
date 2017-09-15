import cv2
import time

from memory_profiler import profile
from memory_profiler import memory_usage
from skimage import exposure
from skimage.feature import hog
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier

BASE = 'AT&T'

if BASE == 'AT&T':
    from k_fold_cv import KFoldCV
else:
    from k_fold_cv_yale import KFoldCV

orientations = 8
pixels_per_cell = (4, 4)
cells_per_block = (1, 1)

n_estimators = 255
max_features = 9
n_jobs = 1
file_pos = '_1'

hog_mean_time = []
train_time = []
predict_mean_time = []
error_per_fold = []
memory_consup = []
memory_predict = []


def get_mean(list):
    sum = 0.0
    for item in list:
        sum += item
    return sum/len(list)


def get_hog_descriptor_and_labels( set, is_train=False, visualise=False):

    samples = []
    targets = []
    hog_time = []

    for index in range(len(set['imgs'])):

        img = set['imgs'][index]
        label = set['labels'][index]

        if visualise:
            hog_descriptor, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                            cells_per_block=cells_per_block, visualise=visualise)
            samples.append(hog_descriptor)
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 100))
            res = resize(hog_image_rescaled, (112 * 3, 96 * 3))
            cv2.imshow('HOG', res)
            cv2.moveWindow('HOG', 0, 0)
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        else:
            hog_time_begin = time.time()
            hog_descriptor = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                            cells_per_block=cells_per_block, visualise=visualise)
            samples.append(hog_descriptor)
            hog_time_end = time.time()
            hog_time.append(hog_time_end-hog_time_begin)

        targets.append(label)

    if is_train:
        print('HOG mean time: {} sec.'.format(get_mean(hog_time)))
        hog_mean_time.append(get_mean(hog_time))

    #print('Descrioptor Size: '+str(len(samples[0])))
    return samples, targets

def reshape_samples( set ):
    #print('Reshape samples and getting labels!!!')
    samples = []
    targets = []

    for index in range(len(set['imgs'])):
        img = set['imgs'][index]
        img = img.reshape(img.shape[0]*img.shape[1])
        label = set['labels'][index]
        samples.append(img)
        targets.append(label)
    #print('Sample size: '+str(len(samples[0])))
    return samples, targets


def get_error(result, targets):
    error = 0.0
    for index in range(len(result)):
        if result[index] != targets[index]:
            error += 1

    return (error/len(result))*100

#@profile
def create_train_random_forest( samples, labels ):

    model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs, verbose=0)

    train_time_begin = time.time()
    #print(memory_usage(model.fit(samples, labels)))
    #model.fit(samples, labels)
    memory_to_train = memory_usage((model.fit, (samples, labels)))
    memory_consup.append(max(memory_to_train)-min(memory_to_train))
    train_time_end =  time.time()
    train_time.append(train_time_end-train_time_begin)
    #del memory_to_train[:]

    print('Training Time: {} sec.'.format(train_time_end - train_time_begin))
    print('Training Memo: {} MIB.'.format(memory_consup[len(memory_consup)-1]))

    return model

#@profile
def test_model(model, test_imgs, targets):

    predict_time = []
    results = []

    for img in test_imgs:
        img = img.reshape(1, -1)
        predict_time_begin = time.time()
        predict_list = model.predict_proba(img)
        result = model.predict(img)
        predict_time_end = time.time()
        predict_time.append(predict_time_end-predict_time_begin)
        #memory_to_predict = memory_usage((model.predict,(img,)))
        #memory_predict.append(max(memory_to_predict)-min(memory_to_predict))
        results.append(result)

    print('Mean prediction time: {} sec.'.format(get_mean(predict_time)))
    predict_mean_time.append(get_mean(predict_time))

    return get_error(results, targets)


print('Random Forest!!!\n')
print('Creating Folds!!!\n')
k_fold_cv = KFoldCV('five_fold_files.txt')
k_fold_cv.create_folds()

print('Random Forest with HOG!!!')
for i in range(len(k_fold_cv.folds)):

    print('\n--------------------------------------------------------')
    print('Getting train and test folds - Test fold {}'.format(i + 1))
    train_test = k_fold_cv.get_train_test_folds(i + 1)

    print('Getting HOG Descriptor to Train Folds!!!')
    samples, labels = get_hog_descriptor_and_labels(train_test['train'], is_train=True, visualise=False )

    print('Training the model!!!')
    model = create_train_random_forest(samples, labels)

    print('Getting HOG Descriptor to Test Folds!!!')
    samples_test, labels_test = get_hog_descriptor_and_labels(train_test['test'])

    print('Testing the model!!!')
    error = test_model(model, samples_test, labels_test)

    print('Error on fold {}: {}%.'.format((i + 1), error))
    error_per_fold.append(error)

print('Mean error: {}%.'.format(get_mean(error_per_fold)))

with open('rf_hog_info{}.txt'.format(file_pos), 'a') as file:
    for index in range(len(k_fold_cv.folds)):
        # file.write('i\n')
        file.write('{}\t{}\t{}\t{}\t{}\n'.format(error_per_fold[index], train_time[index],
                                             predict_mean_time[index], hog_mean_time[index],memory_consup[index]))

del error_per_fold[:]
del train_time[:]
del predict_mean_time[:]
del memory_consup[:]
'''
print('Random Forest without HOG!!!')

if BASE == 'AT&T':
    fold_n = len(k_fold_cv.folds)
else:
    fold_n = len(k_fold_cv.folds) - 1

for i in range(fold_n):

    print('\n--------------------------------------------------------')
    print('Getting train and test folds - Test fold {}'.format(i + 1))
    train_test = k_fold_cv.get_train_test_folds(i + 1)

    print('Getting Samples and Labels of Train Folds!!!')
    samples, labels = reshape_samples(train_test['train'])
    #raw_input('press something!!!')
    print('Training the model!!!')
    model = create_train_random_forest(samples, labels)
    #raw_input('press something!!!')
    #model_size.append(size(model,0))

    print('Getting Samples and Labels of  Test Folds!!!')
    samples_test, labels_test = reshape_samples(train_test['test'])

    print('Testing the model!!!')
    error = test_model(model, samples_test, labels_test)

    print('Error on fold {}: {}%.'.format((i + 1), error))
    error_per_fold.append(error)

print('Mean error: {}%.'.format(get_mean(error_per_fold)))

with open('rf_info{}.txt'.format(file_pos), 'a') as file:
    for index in range(fold_n):
        # file.write('i\n')
        file.write('{}\t{}\t{}\t{}\n'.format(error_per_fold[index], train_time[index],
                                            predict_mean_time[index], memory_consup[index]))#, memory_predict[index]))
        #file.write('{}\t{}\n'.format(memory_consup[index], memory_predict[index]))
        #pass
'''