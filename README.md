
# linear_regression

---

This project lets you learn who to implement a linear regression to predict 
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

Z-score normalisation, also known as standardisation, is a method used to scale 
the values in a dataset so that they have $\mu = 0$ and $\sigma = 1$.
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
            T_step1:Apply Gradient descent
            T_step2:Denormalise thetas
            T_step1 --> T_step2
    }  
    [*] --> train
    train --> [*]
```
#### Gradient descent

Gradient descent is an optimisation method commonly used to adjust the coefficients
of a linear regression model in order to minimise a cost function.

```python
def gradientDescent(data, learningRate, epoch):
    theta0, theta1 = 0, 0 # init thetas
    
    for i in range(epoch): # loop {epoch} time
        _g0 = computeGradient0(data, learningRate, theta0, theta1)
        _g1 = computeGradient1(data, learningRate, theta0, theta1)
        theta0 -= _g0
        theta1 -= _g1
    
    return theta0, theta1 # return thetas

```

computeGradient0 :
```math
    \theta^{}_{0(tmp)} = lr * {1 \over m} * \sum_{i=0}^{m - 1} (estimatePrice(x^{(i)}) − y^{(i)})
```
computeGradient1 :
```math
    \theta^{}_{1(tmp)} =  lr * {1 \over m}  * \sum_{i=0}^{m - 1} (estimatePrice(x^{(i)}) − y^{(i)}) ∗ x^{(i)}
```
Where:
* $lr$ is the learningRate.
* $m$ is the total number of x.
* $estimatePrice()$ the function  $` y^{}_{estimated} = θ^{}_{0} + x * θ^{}_{1} `$

#### denormalise thetas
As thetas are calculated using standardised data, we have to denormalise them to make them match the original data.

##### Standard equations

1. *Normalised data :*
   $$x_{\text{norm}} = \frac{x - \mu_x}{\sigma_x}$$
   $$y_{\text{norm}} = \frac{y - \mu_y}{\sigma_y}$$

2. *Linear regression on normalised data :*
   $$y_{\text{norm}} = \theta_0 + \theta_1 \cdot x_{\text{norm}}$$

##### From $`y_{\text{norm}}`$ to $`y`$

To find $`y`$ from $`y_{\text{norm}}`$, we use the inverse relationship of normalisation :
   $$y = y_{\text{norm}} \cdot \sigma_y + \mu_y$$

Substitute $`y_{\text{norm}}`$ for the standard equation :
   $$y = (\theta_0 + \theta_1 \cdot x_{\text{norm}}) \cdot \sigma_y + \mu_y$$

##### From $`x_{\text{norm}}`$ to $`x`$

For $`x_{\text{norm}}`$, we also use the inverse relationship of normalisation :
   $$x_{\text{norm}} = \frac{x - \mu_x}{\sigma_x} \implies x = x_{\text{norm}} \cdot \sigma_x + \mu_x$$

Substitute $`x_{\text{norm}}`$ in the model equation :
   $$y = \left( \theta_0 + \theta_1 \cdot \frac{x - \mu_x}{\sigma_x} \right) \cdot \sigma_y + \mu_y$$

##### Simplification

Let's develop the equation :
   $$y = \left( \theta_0 \cdot \sigma_y + \theta_1 \cdot \frac{\sigma_y}{\sigma_x} \cdot (x - \mu_x) \right) + \mu_y$$

Let's rearrange the equation to isolate the constant terms and those as a function of $`x`$ :
   $$y = \theta_0 \cdot \sigma_y + \mu_y + \theta_1 \cdot \frac{\sigma_y}{\sigma_x} \cdot x - \theta_1 \cdot \frac{\sigma_y}{\sigma_x} \cdot \mu_x$$

Let's group the constant terms together:
   $$y = \left( \theta_0 \cdot \sigma_y + \mu_y - \theta_1 \cdot \frac{\sigma_y \cdot \mu_x}{\sigma_x} \right) + \theta_1 \cdot \frac{\sigma_y}{\sigma_x} \cdot x$$

##### Identification of denormalised coefficients

By comparing this equation with the standard form $`y = \beta_0 + \beta_1 \cdot x`$, we can identify :
   $$\beta_1 = \theta_1 \cdot \frac{\sigma_y}{\sigma_x}$$
   $$\beta_0 = \theta_0 \cdot \sigma_y + \mu_y - \theta_1 \cdot \frac{\sigma_y \cdot \mu_x}{\sigma_x}$$

### Conclusion

This gives us the denormalisation equations for the coefficients of the linear regression:
   $$\beta_1 = \theta_1 \cdot \frac{\sigma_y}{\sigma_x}$$
   $$\beta_0 = \mu_y + \sigma_y \cdot (\theta_0 - \theta_1 \cdot \frac{\mu_x}{\sigma_x})$$
