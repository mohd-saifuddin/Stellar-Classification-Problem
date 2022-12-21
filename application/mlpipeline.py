import numpy as np
import pandas as pd
import pickle


class Pipline(object):
    """
    This class is a pipeline mechanism to feed the 
    query datapoint into the model for prediction.
    """

    def __init__(self) -> None:
        self.features = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
        self.target = 'class'
        self.df = None
    
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
            _, sig_clf = pickle.load(file=m_pkl)
        
        pred_proba = sig_clf.predict_proba(X=self.df)
        confidence = np.round(a=np.max(a=pred_proba)*100, decimals=2)
        pred_class = sig_clf.predict(X=self.df)[0]
        if pred_class == 'QSO': pred_class = 'Quasi-Stellar Object'
        elif pred_class == 'GALAXY': pred_class = 'Galaxy'
        else: pred_class = 'Star'
        conclusion = "The predicted class is '{}' with a confidence of {}%.".format(pred_class, confidence)
        return conclusion
    
    def ml_pipeline(self):
        """
        This method is a pipeline.
        """
        self.preprocess()
        self.featurize()
        return self.predict()
