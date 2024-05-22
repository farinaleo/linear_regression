
# linear_regression

---

This project lets you learn who to implement a linear regression to predict \
the price of a car  based on its mileage.

---
### Logical structure
```mermaid
stateDiagram
    direction LR
    state Learn {
        direction LR
        extract
        normalise
        train
        [*] --> extract:csv file
        extract --> normalise:dataframe
        normalise--> train:dataframe
    }
    train --> predict:β<sub>0</sub> and β<sub>1</sub>



    state Predict {
        direction LR
        [*] --> predict:value
        predict
        predict --> [*]:predicted value
    }
```
this schema presents the steps to predict a value with data.

---
### Extract Data
```mermaid
stateDiagram
    direction LR
    state extract {
        direction TB
            E_step1:Convert csv to pd.DataFrame
            E_step2:Cast data in float
            E_step3:Find required columns
            
            E_step1 --> E_step2
            E_step2 --> E_step3
    }

    [*] --> extract
    extract --> [*]
```

---
### Normalise Data
```mermaid
stateDiagram
    direction LR
    
    state normalise{
        direction TB
            N_step1:(Apply Z-score Normalisation)
    }
    [*] --> normalise
    normalise --> [*]
```

Z-score normalisation, also known as standardisation, is a method used to scale \
the values in a dataset so that they have $\mu = 0$ and $\sigma = 1$. \
This transformation makes it possible to compare data on different scales.

for X, a list of value, we use :

```math
x^{}_{norm} = {x - \mu^{}_{X} \over \sigma^{}_{X}}
```
where:
*   $\mu^{}_{X}$ is the X mean.
*   $\sigma^{}_{X}$  is the X standard deviation.

---
### Train model
```mermaid
stateDiagram
    direction LR
    state train {
        direction TB
            N_step1:Apply Gradient descent)
    }  
    [*] --> train
    train --> [*]
```

```math
    \theta^{}_{0(tmp)} = lr * {1 \over m} * \sum_{i=0}^{m - 1} (estimatePrice(x^{}_{i}) − y^{}_{i})
```
```math
    \theta^{}_{1(tmp)} =  lr * {1 \over m}  * \sum_{i=0}^{m - 1} (estimatePrice(x^{}_{i}) − y^{}_{i}) ∗ x^{}_{i}
```
