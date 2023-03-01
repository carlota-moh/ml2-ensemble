# from typing import Union
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
# from src.decomposition.seasonal import BaseDecomposition



class OutlierManager:
    def __init__(self):
        pass

    def detect_outlier_sd(self, df, sd_multiple=2,print_=True):
        """
        Detect outliers using standard deviation for each numerical column in a pandas DataFrame.
        A key assumption is the normality of the data. However, in reality, data is rarely normally distributed.
        Hence, this method is not recommended for most cases. Loses the theorethical guarantees quickly.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to detect outliers in.
        sd_multiple : int, optional
            Number of standard deviations to use as the threshold, by default 2
            1 corresponds to 68% confidence interval.
            2 corresponds to 95% confidence interval.
            3 corresponds to 99.7% confidence interval.
        
        print_ : bool, optional
            Whether to print the number of outliers and the percentage of outliers for each column, by default True
        Returns
        -------
        pd.DataFrame
            Boolean mask of outliers for each numerical column.
        """

        # select only the numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns

        # create an empty DataFrame to store the mask for each column
        outlier_masks = pd.DataFrame(columns=numeric_cols, index=df.index)

        # iterate over each numeric column and calculate the outlier mask
        for col in numeric_cols:
            ts = df[col].values
            mean = ts.mean()
            std = ts.std()
            higher_bound = mean + sd_multiple * std
            lower_bound = mean - sd_multiple * std
            outlier_mask = (ts > higher_bound) | (ts < lower_bound)
            if print_:
                print(f"Column: {col}, Nº of Outliers: {outlier_mask.sum()}, % of Outliers: {outlier_mask.mean()*100}%")
            outlier_masks[col] = outlier_mask

        return outlier_masks

    def detect_outlier_iqr(self, df, iqr_multiple=1.5,print_=True):
        """
        Detect outliers using interquartile range.

        Parameters
        ----------
        ts : pd.Series
            Time series to detect outliers in.
        iqr_multiple : int, optional
            Number of interquartile ranges to use as the threshold, by default 1.
            1.5 times the interquartile range is commonly used and corresponds to 2.7 STD almosr 3 STD.
        print_ : bool, optional
            Whether to print the number of outliers and the percentage of outliers for each column, by default True
        Returns
        -------
        pd.DataFrame
            Boolean mask of outliers.
        """
        # select only the numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns

        # create an empty DataFrame to store the mask for each column
        outlier_masks = pd.DataFrame(columns=numeric_cols, index=df.index)
        for col in numeric_cols:
            ts = df[col].values
            q1, q2, q3 = np.quantile(ts, 0.25), np.quantile(ts, 0.5), np.quantile(ts, 0.75)
            iqr = q3 - q1
            higher_bound = q3 + iqr_multiple * iqr
            lower_bound = q1 - iqr_multiple * iqr
            outlier_mask = (ts > higher_bound) | (ts < lower_bound)
            outlier_mask = (ts > higher_bound) | (ts < lower_bound)
            if print_:
                print("Nº de Outliers: ",outlier_mask.sum(), "% de Outliers: ", outlier_mask.mean(), "%")
            outlier_masks[col] = outlier_mask
        return outlier_masks

    def detect_outlier_isolation_forest(self, df, outlier_fraction, print_=True, **kwargs):
        """
        Detect outliers using isolation forest. Is an unsupervised learning algorithm that detects anomalies by isolating outliers.
        Based on Decision Trees. It models the oulliers directly: Assumes that the outlier points fall in the outer periphery and are
        easier to fall in the leaf node of a tree. Therefore, you can find ouliers in the short branches of the tree., whereas normal
        points are more likely to fall in the deeper branches of the tree.

        By default it uses 100 trees, but you can change this by passing the n_estimators parameter to kwargs.

        Parameters
        ----------
        ts : pd.Series
            Time series to detect outliers in.
        outlier_fraction : float, 'auto' or [0, 0.5]
            Fraction of outliers in the time series. Specifies the contamination parameter in IsolationForest.
            Is the same as the contamination parameter in IsolationForest.
        **kwargs
            Keyword arguments to pass to IsolationForest.
        
        Returns
        -------
        pd.Series
            Boolean mask of outliers.
        """
        if isinstance(outlier_fraction, str):
            assert outlier_fraction == "auto", "outlier_fraction must be between 0 and 0.5 or 'auto'"
        else:
            assert outlier_fraction >= 0 and outlier_fraction <= 0.5, "outlier_fraction must be between 0 and 0.5"
        # select only the numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        # create an empty DataFrame to store the mask for each column
        outlier_masks = pd.DataFrame(columns=numeric_cols, index=df.index)

        for col in numeric_cols:
            ts = df[col].values
            min_max_scaler = StandardScaler()
            scaled_time_series = min_max_scaler.fit_transform(ts.reshape(-1, 1))
            kwargs["contamination"] = outlier_fraction
            kwargs["random_state"] = 42
            model = IsolationForest(**kwargs)
            pred = model.fit_predict(scaled_time_series)
            pred = 1 - np.clip(pred, a_min=0, a_max=None)
            outlier_mask = pred.astype(bool)
            if print_:
                print("Nº de Outliers: ",outlier_mask.sum(), "% de Outliers: ", outlier_mask.mean(), "%")
            outlier_masks[col] = outlier_mask       
        return outlier_masks

    # Adapted from https://github.com/nachonavarro/seasonal-esd-anomaly-detection
    def calculate_test_statistic(self, ts, hybrid=False):
        """
        Calculate the test statistic for ESD.
        Calculate the test statistic defined by being the top z-score

        Parameters
        ----------
        ts : np.array
            Time series to calculate the test statistic for.
        hybrid : bool, optional
            Whether to use the hybrid ESD test statistic, by default False

        Returns
        -------
        int
            Index of the test statistic.
        float
            Value of the test statistic.
        """
        if hybrid:
            median = np.ma.median(ts)
            mad = np.ma.median(np.abs(ts - median))
            scores = np.abs((ts - median) / mad)
        else:
            scores = np.abs((ts - ts.mean()) / ts.std())
        max_idx = np.argmax(scores)
        return max_idx, scores[max_idx]

    def calculate_critical_value(self, size, alpha):
        """
        Calculate the critical value for ESD.
        https://en.wikipedia.org/wiki/Grubbs%27_test_for_outliers#Definition

        Parameters
        ----------
        size : int
            Size of the time series.
        alpha : float
            Significance level.
        
        Returns
        -------
        float
            Critical value.
        """

        t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
        numerator = (size - 1) * t_dist
        denominator = np.sqrt(size ** 2 - size * 2 + size * t_dist ** 2)
        return numerator / denominator

    # def seasonal_esd(
    #     self,
    #     ts: Union[pd.DataFrame, pd.Series],
    #     seasonal_decomposer: BaseDecomposition,
    #     hybrid: bool = False,
    #     max_anomalies: int = 10,
    #     alpha: float = 0.05,
    # ):
    #     """
    #     Seasonal ESD for detecting outliers in time series.

    #     Parameters
    #     ----------
    #     ts : Union[pd.DataFrame, pd.Series]
    #         Time series to detect outliers in.
    #     seasonal_decomposer : BaseDecomposition
    #         Seasonal decomposition model.
    #     hybrid : bool, optional
    #         Whether to use the hybrid ESD test statistic, by default False
    #     max_anomalies : int, optional
    #         Maximum number of anomalies to detect, by default 10
    #     alpha : float, optional
    #         Significance level, by default 0.05
        
    #     Returns
    #     -------
    #     pd.Series
    #         Boolean mask of outliers.
    #     """

    #     if max_anomalies >= len(ts) / 2:
    #         raise ValueError(
    #             "The maximum number of anomalies must be less than half the size of the time series."
    #         )

    #     decomposition = seasonal_decomposer.fit(ts)
    #     # Checking if MultiSeasonalDecomposition
    #     # if hasattr(seasonal_decomposer, "seasonal_model"):
    #     #     seasonal = np.sum(list(decomposition.seasonal.values()), axis=0)
    #     # else:
    #     seasonal = decomposition.total_seasonality
    #     residual = ts - seasonal - np.median(ts)
    #     outliers = self.generalized_esd(
    #         residual, max_anomalies=max_anomalies, alpha=alpha, hybrid=hybrid)
    #     return outliers
    
    def detect_outlier_generalized_esd(self, df, max_anomalies=10, alpha=0.05, hybrid=False, print_=True):
        """
        Generalized ESD (Extreme Studentized Deviate) for detecting outliers in time series. More sophisticated than the STD but still uses 
        the same assumptions of normality. Its based om Grubbs statistical test, which is used to find a single
        outlier in normally distributed data. ESD iteratively removes the most extreme value and recalculates
        the critical value based on the remaining data.

        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

        Parameters
        ----------
        ts : Union[pd.DataFrame, pd.Series]
            Time series to detect outliers in.
        max_anomalies : int, optional
            Maximum number of anomalies to detect, by default 10.
            The number of times the Grubbs' Test will be applied to the ts.
        alpha : float, optional
            Significance level, by default 0.05
        hybrid : bool, optional
            Whether to use the hybrid ESD test statistic, by default False
            A flag that determines the type of z-score.
        
        Returns
        -------
        pd.Series
            Boolean mask of outliers.
        """
        # select only the numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns

        # create an empty DataFrame to store the mask for each column
        outlier_masks = pd.DataFrame(columns=numeric_cols, index=df.index)

        for col in numeric_cols:
            ts = df[col].values
            ts = np.ma.array(
                ts
            )
            test_statistics = []
            num_outliers = 0
            for curr in range(max_anomalies):
                test_idx, test_val = self.calculate_test_statistic(ts, hybrid=hybrid)
                critical_val = self.calculate_critical_value(len(ts) - curr, alpha)
                if test_val > critical_val:
                    num_outliers = curr
                test_statistics.append(test_idx)
                ts[
                    test_idx
                ] = (
                    np.ma.masked
                )
            anomalous_indices = test_statistics[: num_outliers + 1] if num_outliers > 0 else []
            outlier_mask = np.zeros_like(ts)
            outlier_mask[anomalous_indices] = 1
            outlier_mask = outlier_mask.astype(bool)
            if print_:
                print("Nº de Outliers: ",outlier_mask.sum(), "% de Outliers: ", outlier_mask.mean(), "%")
            outlier_masks[col] = outlier_mask   
        return outlier_masks
    

    
    def detect_outlier_multivariable_isolation_forest(self,df,contaminacion=0.01,n_estimators=500,plot=True,random_state=42):
        """
        Isolation Forest for detecting outliers in multivariate time series. It is based on the idea of isolating outliers in a dataset by randomly
        selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to detect outliers in.
        contaminacion : float, optional
            The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
            Used when fitting to define the threshold on the decision function. by default 0.01
        n_estimators : int, optional
            The number of base estimators in the ensemble. by default 100
        plot : bool, optional
            Whether to plot the results, by default True
        
        Returns
        -------
        pd.DataFrame
            Dataframe with the outliers detected.
        """
        #obatain the numeric columns of the dataframe
        numeric_cols = df.select_dtypes(include=np.number).columns
        #Isolation Forest
        ISO_FOREST=IsolationForest(n_estimators=n_estimators,
                    max_samples=0.4, 
                    contamination=contaminacion, 
                    max_features=1.0, 
                    bootstrap=True, 
                    n_jobs=-1, 
                    random_state=random_state, 
                    verbose=0, 
                    warm_start=False)
        #fit the model
        ISO_FOREST.fit(df[numeric_cols])
        outliers=ISO_FOREST.predict(df[numeric_cols])
        df_anomaly=df[numeric_cols].copy()
        df_anomaly[f'MULTIVARIABLE_ISO_FOREST_{contaminacion}']=outliers
        #print the percentage of outliers
        print("Nº de Outliers: ",df_anomaly[f'MULTIVARIABLE_ISO_FOREST_{contaminacion}'].value_counts()[-1], ",  % de Outliers: ",
               df_anomaly[f'MULTIVARIABLE_ISO_FOREST_{contaminacion}'].value_counts(normalize=True)[-1]*100, "%")
        if plot: #plot the results
            print(f"The PCA is only for plotting purposes, the model is not trained with it. The model is trained with the original data. Model arguments are: contamination={contaminacion}, n_estimators=500, max_samples=0.4, max_features=1.0, bootstrap=True, n_jobs=-1, random_state={random_state}, verbose=0, warm_start=False")
            scaler = StandardScaler()
            X_transformed = scaler.fit_transform(X=df[numeric_cols])
            pca = PCA(n_components=3,) #Tanatas compoentes como variables tienen los datos y ya posteriormente elegimos las optimas
            X_pca = pca.fit_transform(X_transformed)
            exp_variance = pd.DataFrame(data=pca.explained_variance_ratio_, index = ['PC' + str(n_pca + 1) for n_pca in range(pca.n_components)], columns=['Exp_variance'])
            exp_variance['cum_Exp_variance'] = exp_variance['Exp_variance'].cumsum()
            fig = go.Figure(data=[go.Scatter3d(x=X_pca[:,0], y=X_pca[:,1], z=X_pca[:,2], mode='markers', marker=dict(color=outliers, size=3, opacity=0.8, colorscale='viridis'))])
            fig.update_layout(title='Isolation Forest', scene = dict(
                            xaxis_title='PC1',
                            yaxis_title='PC2',
                            zaxis_title='PC3'),
                            width=1400,
                            margin=dict(r=1, l=5, b=5, t=40))
            #add title to the plot and legend of anomaly
            fig.update_layout(title_text="Isolation Forest", title_x=0.5)
            #annotate the % of variance explained by each PC3 out of the 3D
            fig.add_annotation(x=0.5, y=1, text=f"Three principal components represents a variance of %"+ str(round(exp_variance.cum_Exp_variance["PC3"]*100,4))
                            +" of original data",
                                showarrow=False,
                                font=dict(size=16, color="black"))

            #add legend of anomaly color
            fig.show()

        #return an outlier mask
        return (df_anomaly[f'MULTIVARIABLE_ISO_FOREST_{contaminacion}'].apply(lambda x: True if x==-1 else False)).to_frame()
    
    def summary(self, df,iqr_multiple=1.5,sd_multiple=3,isolation_outlier_fraction='auto',generalized_esd={"max_anomalies":10, "alpha":0.05, "hybrid":False}):
        #IQR
        outliers_iqr=self.detect_outlier_iqr(df,iqr_multiple=iqr_multiple,print_=False)
        percentage_outliers_iqr=outliers_iqr.mean()*100
        #SD
        outliers_sd=self.detect_outlier_sd(df,sd_multiple=sd_multiple,print_=False)
        percentage_outliers_sd=outliers_sd.mean()*100
        #Isolation Forest
        outliers_isolation_forest=self.detect_outlier_isolation_forest(df,outlier_fraction=isolation_outlier_fraction,print_=False)
        percentage_outliers_isolation_forest=outliers_isolation_forest.mean()*100
        #Generalized ESD
        outliers_generalized_esd=self.detect_outlier_generalized_esd(df,max_anomalies=generalized_esd["max_anomalies"],
                                                                       alpha=generalized_esd["alpha"],hybrid=generalized_esd["hybrid"],print_=False)
        percentage_outliers_generalized_esd=outliers_generalized_esd.mean()*100

        #obatain the numeric columns of the dataframe
        numeric_cols = df.select_dtypes(include=np.number).columns

        #make a mutiindex dataframe of the methods and the columns variables of the df
        index = pd.MultiIndex.from_product([numeric_cols,['% Percentage Outliers','INDEX']])
        columns=[f'IQR_{iqr_multiple}', f'STD_{sd_multiple}', f'ISO FOREST_{isolation_outlier_fraction}', f'GEN ESD_{generalized_esd["max_anomalies"]}_{generalized_esd["alpha"]}']
        summary_ = pd.DataFrame(index=index, columns=columns)
        for col in numeric_cols:
            #fill the dataframe with the values of the previous methods 
            #IQR
            summary_.loc[(col, '% Percentage Outliers'),f'IQR_{iqr_multiple}'] = percentage_outliers_iqr[col]
            summary_.loc[(col, 'INDEX'),f'IQR_{iqr_multiple}'] = outliers_iqr[col][outliers_iqr[col]==True].index
            #SD
            summary_.loc[(col, '% Percentage Outliers'),f'STD_{sd_multiple}'] = percentage_outliers_sd[col]
            summary_.loc[(col, 'INDEX'),f'STD_{sd_multiple}'] = outliers_sd[col][outliers_sd[col]==True].index
            #Isolation Forest
            summary_.loc[(col, '% Percentage Outliers'),f'ISO FOREST_{isolation_outlier_fraction}'] = percentage_outliers_isolation_forest[col]
            summary_.loc[(col, 'INDEX'),f'ISO FOREST_{isolation_outlier_fraction}'] = outliers_isolation_forest[col][outliers_isolation_forest[col]==True].index
            #Generalized ESD
            summary_.loc[(col, '% Percentage Outliers'),f'GEN ESD_{generalized_esd["max_anomalies"]}_{generalized_esd["alpha"]}'] = percentage_outliers_generalized_esd[col]
            summary_.loc[(col, 'INDEX'),f'GEN ESD_{generalized_esd["max_anomalies"]}_{generalized_esd["alpha"]}'] = outliers_generalized_esd[col][outliers_generalized_esd[col]==True].index
        summary_.style.apply(lambda x: ['background: red' if x.name[1]=='% Percentage Outliers' and v > 5 else '' for v in x], axis=1)
        
        return summary_

