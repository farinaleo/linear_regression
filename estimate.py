from srcs.estimatedPrice import estimated_price


def main():
    try:
        mileage = int(input("Please enter your car mileage: "))

        estimate = estimated_price(mileage)
        print(f"Our super calculator predict a price of: {estimate}")

        if estimate == 0:
            print("The model is probably not trained. Please launch <python train.py> before using estimate.")
        elif estimate < 0:
            print("LMAO I have never seen a car like that. Please throw it.")
            print("The prediction is a simple linear model. As the predicted price is inversely proportional to the "
                  "mileage, the prediction tends towards minus infinity.")
        elif mileage < 0:
            print("You gave a negative mileage. As the prediction is a linear equation we can find a price but please "
                  "understand that the model is really simple and in reality you can't predict a price with a "
                  "negative mileage.")
    except Exception:
        print("Impossible to estimate the price. Please Make sure your mileage is correctly formatted \
(supposed as an int greater than zero).")


if __name__ == '__main__':
    main()
