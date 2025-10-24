from src.data_loader import load_data
from src.model_builder import build_model
from src.trainer import train_model
from src.webcam_capture import capture_image

def main():
    print("Loading dataset...")
    dataset_path = "data"
    X_train, y_train, X_test, y_test = load_data(dataset_path)

    print("Building model...")
    model = build_model()

    print("Training model...")
    train_model(model, X_train, y_train, X_test, y_test)

    print("Launching webcam for prediction...")
    capture_image(model)

if __name__ == "__main__":
    main()
