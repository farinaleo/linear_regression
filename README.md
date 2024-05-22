
# linear_regression

---

This project lets you learn who to implement a linear regression to predict \
the price of a car  based on its mileage.

---
### Logical structure
```mermaid
flowchart LR
    subgraph input["Input"]
    end
    
    subgraph extract["Extract Data"]
    end
    
    subgraph normalise["Normalise Data"]
    end
    
    subgraph train["Train model"]
    end
    
    subgraph predict["Predict"]
    end
    
    input -- csv file --> extract
    extract -- dataFrame --> normalise
    normalise --> train
    train -- β<sub>0</sub> and β<sub>1</sub> --> predict
```
this schema presents the steps to predict a value with data.

---
### Extract Data
```mermaid
flowchart LR
    subgraph input
    end
    
    subgraph extract["Extract data"]
        direction TB
            E_step1(Convert csv to pd.DataFrame)
            E_step2(Cast data in float)
            E_step3(Find required columns)
            
            E_step1 --> E_step2 --> E_step3
    end
    
    subgraph output
    end
    
    input --> extract --> output
```

---
### Normalise Data
```mermaid
flowchart LR
    subgraph input
    end
    
    subgraph normalise["Normalise data"]
        direction TB
            N_step1(Apply Z-score Normalisation)
    end
    
    subgraph output
    end
    
    input --> normalise --> output
```

Z-score normalisation, also known as standardisation, is a method used to scale \
the values in a dataset so that they have $\mu = 0$ and $\sigma = 1$. \
This transformation makes it possible to compare data on different scales. \

we use :

```math
x^{}_{norm} = {x - \mu^{}_{x} \over \sigma^{}_{x}}
```
where:
*   $\mu^{}_{x}$ is the x set mean.
*   $\sigma^{}_{x}$  is the x set standard deviation.
