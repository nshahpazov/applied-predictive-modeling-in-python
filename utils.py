def eval(name, model):
    return pd.DataFrame({
        "rmse": cross_val_score(model, X_train, y_train, cv=5, scoring=rmse_scorer),
        "name": name,
        "fold": range(1, 6),
    })

def predict(name, model, X):
    return X.assign(y=model.predict(X), name=name)

