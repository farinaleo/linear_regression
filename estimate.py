from srcs.estimatedPrice import estimatedPrice


def main():
    try:
        # user input
        mileage = int(input("Please enter your car mileage: "))
        # estimate price
        estimate = estimatedPrice(mileage)
        print(f"Our super calculator predict a price of: {estimate}")

        # handle error
        if estimate == 0:
            print("The model is probably not trained. Please launch <python train.py> before using estimate.")
        elif estimate < 0:
            print("LMAO I have never seen a car like that. Please throw it.")
    except Exception:
        print("Impossible to estimate the price. Please Make sure your mileage is correctly formatted \
(supposed as an int greater than zero).")


if __name__ == '__main__':
    main()
