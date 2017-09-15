import cv2
from skimage.transform import resize


class KFoldCV:

    def __init__(self, file_name):
        self.file_name = file_name
        self.folds = []

    def create_folds(self):
        #print("Creating Folds!!!")

        with open(self.file_name, 'r') as f:

            for line in f:

                fold_aux = line.split('/')
                fold_aux = fold_aux[:-1]
                subjects = {}
                imgs = []
                labels = []

                for i in range(1, len(fold_aux) + 1):

                    subj_ids = fold_aux[i - 1].split(',')

                    for id in subj_ids:

                        folder_name = 's' + str(i)
                        img_name = id + '.pgm'
                        img = cv2.imread('orl_faces/' + folder_name + '/' + img_name, 0)
                        # read image as grayscale
                        imgs.append(img)
                        labels.append(i)
                        '''
                        img = resize(img, (112 * 3, 96 * 3))
                        cv2.imshow('Faces', img)
                        cv2.moveWindow('Faces', 0, 0)
                        cv2.waitKey(150)
                        cv2.destroyAllWindows()
                        # imgs.append(folder_name + '/' + img_name)
                        '''

                subjects['imgs'] = imgs
                subjects['labels'] = labels
                self.folds.append(subjects)

    def get_train_test_folds(self, test_fold_number):
        #print('Getting Training and test folds!!!')

        train_imgs = []
        train_labels = []
        test_imgs = []
        test_labels = []

        train = {}
        test = {}
        train_test = {}

        for i in range(len(self.folds)):
            if i != test_fold_number - 1:
                train_imgs += self.folds[i]['imgs']
                train_labels += self.folds[i]['labels']
            else:
                test_imgs = self.folds[i]['imgs']
                test_labels = self.folds[i]['labels']

        train['imgs'] = train_imgs
        train['labels'] = train_labels
        test['imgs'] = test_imgs
        test['labels'] = test_labels

        train_test['train'] = train
        train_test['test'] = test

        return train_test
