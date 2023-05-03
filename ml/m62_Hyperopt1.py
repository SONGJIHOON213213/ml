# 최소값을 찾는거다.
# BayesianOptimization 최대값 찾는거다.
import hyperopt 
import numpy as np
import pandas as pd
print(hyperopt.__version__) # 0.2.7
from hyperopt import hp, fmin, tpe, Trials

search_space = {'x1' : hp.quniform('x', -10, 10, 1),
                'x2' : hp.quniform('x2', -15, 15, 1)}
                     # hp.quniform(label, low, high, q)
print(search_space)
# {'x1': <hyperopt.pyll.base.Apply object at 0x0000024E2F240070>, 'x2': <hyperopt.pyll.base.Apply object at 0x0000024E34DB3BE0>}

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1 **2 -20 * x2
    
    return return_value

trial_val = Trials()

best = fmin(fn = objective_func,
            space = search_space,
            algo = tpe.suggest, # 디폴트 
            max_evals = 30                                                                                                                                                              , # n_iter와 동일
            trials = trial_val,
            rstate = np.random.default_rng(seed = 1234))
print('best : ', best) # best :  {'x': -2.0, 'x2': 14.0}
# best :  {'x': 0.0, 'x2': 15.0}
print(trial_val.results)
print(trial_val.vals)

############# pandas 데이터프레엠이 trial_val를 넣어라 ################

abc = pd.DataFrame(trial_val.vals)
print(abc)