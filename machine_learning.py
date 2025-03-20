from sklearn.tree import DecisionTreeRegressor;
from sklearn.metrics import mean_absolute_error;


get_model=DecisionTreeRegressor()

def executeModel (x, y):
    fit_model=get_model.fit(x, y)
    x_head=x.head()
    y_head=y.head()
    print_base_info(x_head, y_head)
    print(fit_model.predict(x_head))
    
def detect_error_result(x, y):
    predict_prices=get_model.predict(x)
    mean_absolute_error(y, predict_prices)


def print_base_info(x_head, y_head):
    print("predicciones para las 5 primeras casas")
    print(x_head)
    print("el precio real es:")
    print(y_head)
    print("Las predicciones son")
