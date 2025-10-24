def train_model(model, X_train, y_train, X_test, y_test, epochs=5):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
    model.save("gesture_model.keras")
    return history
