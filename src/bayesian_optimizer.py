from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.model_selection import cross_val_score


class ModelOptimizer:
    """
    Class used for hyperparameter tuning based on Bayesian Optimization

    Functions based on code from: https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec
    """
    def __init__(self, scoring):
        self.optimizer = None
        self.best_optimizer = None
        self.scoring = scoring

    def black_box_function(self, X_train_scale, y_train, model, **params):
        """
        Black box function for optimization algorithm
        """
        model = model.set_params(**params)
        f = cross_val_score(model, X_train_scale, y_train,
                            scoring=self.scoring, cv=10)
        
        return f.mean()

    def optimize_model(self, pbounds, X_train_scale, y_train, model, int_params, n_iter=25):
        """
        Optimize 

        """
        def opt_function(**params):
            """
            Function wrapper
            """
            return self.black_box_function(X_train_scale,
                                           y_train,
                                           model,
                                           **params)
        # create optimizer
        optimizer = BayesianOptimization(f = None,
                                         pbounds=pbounds,
                                         verbose=2,
                                         random_state=2022)

        # declare acquisition function used for getting new values of the
        # hyperparams
        utility = UtilityFunction(kind = "ucb", kappa = 1.96, xi = 0.01)

        # Optimization for loop.
        for i in range(n_iter):
            # Get optimizer to suggest new parameter values to try using the
            # specified acquisition function.
            next_point = optimizer.suggest(utility)
            # Force degree from float to int.
            for int_param in int_params:
                next_point[int_param] = int(next_point[int_param])
            # Evaluate the output of the black_box_function using 
            # the new parameter values.
            target = opt_function(**next_point)
            try:
                # Update the optimizer with the evaluation results. 
                # This should be in try-except to catch any errors!
                optimizer.register(params = next_point, target = target)
            except:
                pass
                   
        print("Best result: {}.".format(optimizer.max["params"]))

        self.optimizer = optimizer
        self.best_optimizer = optimizer.max

        return optimizer.max["params"]