import cv2
import numpy as np
import time

#from memory_profiler import profile

BASE = 'AT&T'

if BASE == 'AT&T':
    from k_fold_cv import KFoldCV
else:
    from k_fold_cv_yale import KFoldCV

eigenfaces_n = 40
error_per_fold = []
train_time = []
predict_time = []
memory_consup = []
memory_predict = []
fold_n = 0

def get_model_info(model, high, width, save_imgs =  None):
    print('Model Info!!!')
    print("Model - getNumComponents():"+str(model.getNumComponents()))
    print("Model - getThreshold():" + str(model.getThreshold()))

    if save_imgs:
        mean = model.getMean()
        mean = mean.reshape(high, width)
        cv2.imwrite('mean_image.png', mean)

        eigenvectors = model.getEigenVectors()
        eigenfaces = []
        for index in range(model.getNumComponents()):
            for ei_aux in eigenvectors:
                eigenfaces.append(ei_aux[index])
            a = np.array(eigenfaces)

            #normalize colors
            a = np.interp(a, [eigenvectors.min(), eigenvectors.max()], [0, 255])

            cv2.imwrite('eigenfaces/eigenface_'+str(index)+'.png', a.reshape(high, width))
            eigenfaces = []


def get_mean(list):
    sum = 0.0
    for item in list:
        sum += item

    return (sum/len(list))

@profile
def create_train_model(n_eingenfaces, train_fold, threshold = 0):
    model = None

    if threshold > 0:
        model = cv2.face.createEigenFaceRecognizer(n_eingenfaces, threshold)
    else:
        model = cv2.face.createEigenFaceRecognizer(n_eingenfaces)

    train_time_begin = time.time()
    model.train(train_fold['imgs'], np.array(train_fold['labels']))
    #print(memory_usage((model.train, (train_fold['imgs'], np.array(train_fold['labels'])))))
    #memory_to_trian = memory_usage((model.train, (train_fold['imgs'], np.array(train_fold['labels']))))
    #memory_consup.append(max(memory_to_trian)-min(memory_to_trian))
    train_time_end = time.time()
    train_time.append(train_time_end-train_time_begin)

    print('Training Time: {} sec.'.format(train_time_end-train_time_begin))

    return model

@profile
def test_model(model, test_fold):

    error = 0.0
    predict_time_aux = []
    for i in range(len(test_fold['imgs'])):

        predict_time_begin = time.time()
        label_predicted, conf = model.predict(test_fold['imgs'][i])
        predict_time_end = time.time()
        predict_time_aux.append(predict_time_end-predict_time_begin)

        #memory_to_predict = memory_usage((model.predict, (test_fold['imgs'][i],)))
        #memory_predict.append(max(memory_to_predict)-min(memory_to_predict))

        label_actual = test_fold['labels'][i]

        if label_predicted != label_actual:
            error += 1

    print('Mean prediction time: {} milliseconds.'.format(get_mean(predict_time_aux)*1000))
    predict_time.append(get_mean(predict_time_aux))

    return (error/len(test_fold['imgs']))*100


def save_info():
    with open('eigenface_info.txt', 'a') as file:
        for index in range(fold_n):
            #file.write('i\n')
            file.write('{}\t{}\t{}\t{}\n'.format(error_per_fold[index], train_time[index],
                                                 predict_time[index], memory_consup[index]))
	        #,memory_predict[index]))
            #file.write('{}\t{}\n'.format(memory_consup[index], memory_predict[index]))

print('Eigenfaces!!!\n')
print('Creating Folds!!!')
k_fold_cv = KFoldCV('five_fold_files.txt')
k_fold_cv.create_folds()

if BASE == 'AT&T':
    fold_n = len(k_fold_cv.folds)
else:
    fold_n = len(k_fold_cv.folds) - 1


for i in range(fold_n):
    print('\n--------------------------------------------------------')
    print('Getting train and test folds - Test fold {}'.format(i+1))
    train_test = k_fold_cv.get_train_test_folds(i+1)
    #raw_input('press something!!!')
    print('Training the model!!!')
    model = create_train_model(eigenfaces_n, train_test['train'])
    #raw_input('press something!!!')
    #model_size.append(size(model, 0))

    print('Testing the model!!!')
    error = test_model(model, train_test['test'])

    print('Error on fold {}: {}%.'.format((i+1), error))
    error_per_fold.append(error)

print('Mean error: {}%.'.format(get_mean(error_per_fold)))
#get_model_info(model, 112, 92)
#save_info()

#input('press something!!!')
