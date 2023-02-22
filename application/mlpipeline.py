import warnings
warnings.filterwarnings('ignore')

import cv2 as cv
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import random

class Pipeline(object):
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
    
    def predict(self) -> tuple:
        """
        This method predicts the query datapoint.
        """
        with open(file='stacking_classifier.pkl', mode='rb') as m_pkl:
            clf  = pickle.load(file=m_pkl)
        
        pred_proba = clf.predict_proba(X=self.df)
        confidence = np.round(a=np.max(a=pred_proba)*100, decimals=2)
        pred_class = clf.predict(X=self.df)[0]

        fig = self.get_encoded_image(pred_class=pred_class)

        if pred_class == 'QSO':
            pred_class = 'Quasi-Stellar Object'
        elif pred_class == 'GALAXY':
            pred_class = 'Galaxy'
        else:
            pred_class = 'Star'

        c = "The predicted class is '{}' with a confidence of {}%.".format(pred_class, confidence)
        return c, fig, pred_class
    
    def get_encoded_image(self, pred_class):
        """
        This method selects an image and encodes it.
        """
        image_root = os.path.join(os.getcwd(), './images')
        if pred_class == 'QSO':
            image_class_path = os.path.join(image_root, 'qsos')
        elif pred_class == 'GALAXY':
            image_class_path = os.path.join(image_root, 'galaxies')
        else:
            image_class_path = os.path.join(image_root, 'stars')
        image_picked = random.choice(seq=os.listdir(path=image_class_path))
        image_picked_path = os.path.join(image_class_path, image_picked)

        image = cv.imread(filename=image_picked_path)
        image = cv.cvtColor(src=image, code=cv.COLOR_BGR2RGB)

        image_fig = px.imshow(img=image)
        image_fig.update_layout(
            coloraxis_showscale=False,
            autosize=True, height=500,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        image_fig.update_xaxes(showticklabels=False)
        image_fig.update_yaxes(showticklabels=False)
        return image_fig

    def pipeline(self) -> str:
        """
        This method is a pipeline.
        """
        self.fetch()
        self.preprocess()
        self.featurize()
        return self.predict()
