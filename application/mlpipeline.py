import warnings
warnings.filterwarnings('ignore')

import base64
import numpy as np
import os
import pandas as pd
import pickle
import random


class Pipline(object):
    """
    This class is a pipeline mechanism to feed the 
    query datapoint into the model for prediction.
    """

    def __init__(self, data) -> None:
        self.data = data
        self.features = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
        self.target = 'class'
        self.df = None
    
    def fetch(self) -> None:
        """
        This method fetches the data.
        """
        self.df = pd.DataFrame(data=self.data, columns=self.features)

    def preprocess(self) -> None:
        """
        This method preprocesses the data.
        """
        with open(file='scaling.pkl', mode='rb') as s_pkl:
            scaling = pickle.load(file=s_pkl)
        
        self.df = scaling.transform(X=self.df)
        self.df = pd.DataFrame(data=self.df, columns=self.features)
    
    def featurize(self) -> None:
        """
        This method performs feature engineering on the data.
        """
        fi_cols = ['redshift', 'g-r', 'i-z', 'u-r', 'i-r', 'z-r', 'g']
        self.df['g-r'] = self.df['g'] - self.df['r']
        self.df['i-z'] = self.df['i'] - self.df['z']
        self.df['u-r'] = self.df['u'] - self.df['r']
        self.df['i-r'] = self.df['i'] - self.df['r']
        self.df['z-r'] = self.df['z'] - self.df['r']
        self.df = self.df[fi_cols]
    
    def predict(self) -> str:
        """
        This method predicts the query datapoint.
        """
        with open(file='gb_classifier.pkl', mode='rb') as m_pkl:
            _, sig_clf, _  = pickle.load(file=m_pkl)
        
        pred_proba = sig_clf.predict_proba(X=self.df)
        confidence = np.round(a=np.max(a=pred_proba)*100, decimals=2)
        pred_class = sig_clf.predict(X=self.df)[0]

        encoded_image = self.get_encoded_image(pred_class=pred_class)

        if pred_class == 'QSO':
            pred_class = 'Quasi-Stellar Object'
        elif pred_class == 'GALAXY':
            pred_class = 'Galaxy'
        else:
            pred_class = 'Star'

        c = "The predicted class is '{}' with a confidence of {}%.".format(pred_class, confidence)
        return c, encoded_image
    
    def get_encoded_image(self, pred_class):
        """
        This method selects an image and encodes it.
        """
        image_root = os.path.join(os.getcwd(), './images')
        if pred_class == 'QSO':
            # image_class_path = os.path.join(image_root, 'qsos')
            image_class_path = os.path.join(image_root, 'stars')
        elif pred_class == 'GALAXY':
            # image_class_path = os.path.join(image_root, 'galaxies')
            image_class_path = os.path.join(image_root, 'stars')
        else:
            image_class_path = os.path.join(image_root, 'stars')
        image_picked = random.choice(seq=os.listdir(path=image_class_path))
        image_picked_path = os.path.join(image_class_path, image_picked)

        image_picked_path = open(file=image_picked_path, mode='rb').read()
        encoded_image = base64.b64encode(s=image_picked_path)
        encoded_image = encoded_image.decode()
        return 'data:image/jpeg;base64,{}'.format(encoded_image)
        
    def pipeline(self) -> str:
        """
        This method is a pipeline.
        """
        self.fetch()
        self.preprocess()
        self.featurize()
        return self.predict()
