from inference.predictor import Predictor


def main():

    print("AI Sentiment Engine Started")

    predictor = Predictor()

    while True:

        text = input("\nEnter text (type 'exit' to stop): ")

        if text.lower() == "exit":
            print("Exiting AI Engine...")
            break

        result = predictor.predict(text)

        print("Prediction:", result)


if __name__ == "__main__":
    main()