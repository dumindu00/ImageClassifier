# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import cv2
# import os


# class Model:
#     def __init__(self, img_size=(64, 64), num_classes=2):
        
#         self.img_size = img_size
#         self.num_classes = num_classes
#         self.model = self.build_model()
        
#     def build_model(self):
#         model = Sequential([
#             Conv2D(32, (3,3), activation='relu', input_shape=(*self.img_size, 1)),
#             MaxPooling2D(2,2),
#             Conv2D(64, (3, 3), activation='relu'),
#             MaxPooling2D(2, 2),
#             Flatten(),
#             Dense(64, activation='relu'),
#             Dense(self.num_classes, activation='softmax')
#         ])
        
#         model.compile(optimizer='adam',
#                     loss='sparse_categorical_crossentropy',
#                     metrics=['accuracy']
#                     )
        
#         return model
    
#     # Preprocess Image
    
#     def preprocess_image(self, img_path):
#         img = cv2.imread(img_path, cv2.IMREaD_GRAYSCALE)
#         img = cv2.resize(img, self.img_size)
#         img = img.astype('float') / 255.0   # Normalize the pixel values
#         return img.reshape(*self.img_size, 1)
        
        
        
#         # Load Data
#     def load_data(self):
#         X, y = [], []
#         for class_num in range(1, self.num_classes + 1):
#             folder = str(class_num)
#             for filename in os.listdir(folder):
#                 path = os.path.join(folder, filename)
#                 X.append(self.preprocess_image(path))
#                 y.append(class_num - 1)
        
#         return np.array(X), np.array(y)
    
    
#     # Train Model
#     def train_model(self, counters):
#         X, y = self.load_data()
#         self.model.fit(X, y, epochs=10, batch_size=8, verbose=1)
#         self.model.save('cnn_model.h5')
    
    
#     # Predict Image
#     def predict_image(self, frame):
#         img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         img = cv2.resize(img, self.img_size)
#         img = img.astype('float32') / 255.0
#         img = img.reshape(1, *self.img_size, 1)
#         pred = self.model.predict([img])
#         return np.argmax(pred)



from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL


class Model:
    
    def __init__(self):
        self.model = LinearSVC()
        
    def train_model(self, counters):
        img_list = np.array([])
        class_list = np.array([])
    
        for i in range(i, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg')[:,:,0]
            img = img.reshape(16800)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)
            
        for i in range(i, counters[i]):
            img = cv.imread(f'2/frames{i}.jpg')[:,:,0]
            img = img.reshape(16800)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)
            
        img_list = img_list.reshape(counters[0] - 1 + counters[i] - 1, 16800)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")



    def predict(self, frame):
        frame = frame[1]
        cv.imwrite('frame.jpg', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open('frame.jpg')
        img.thumbnail((150, 150), PIL.Image.Resampling.LANCZOS)
        img.save('frame.jpg')
        
        img = cv.imread('frame.jpg')[:,:,0]
        img = img.reshape(16800)
        prediction = self.model.predict([img])
        
        return prediction[0]
