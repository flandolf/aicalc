import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init
from datetime import datetime

# Initialize colorama for cross-platform colored text
init()

# Initialize calculation history
history = []

# Create and train the model
def create_and_train_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(2,))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                  loss='mean_squared_error')
    
    # Generate training data between 0-1 (model works best in this range)
    X_train = np.random.uniform(0, 100, (10000, 2))  # Generates values between 0 and 100
    y_train = X_train.sum(axis=1)
    
    # Custom training loop to show progress
    print(f"\n{Fore.YELLOW}Training the model...{Style.RESET_ALL}")
    epochs = 100
    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        print(f"{Fore.CYAN}Epoch {epoch + 1}/{epochs} - loss: {loss:.6f}{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}Training complete!{Style.RESET_ALL}")
    return model

# Calculate confidence score (based on prediction error)
def calculate_confidence(prediction, actual):
    error = abs(prediction - actual)
    if actual == 0:
        return 100.0 if error < 1e-6 else 0.0
    accuracy = (1 - error/actual) * 100
    return max(0.0, min(100.0, accuracy))

# Print colored help message
def print_help():
    print(f"\n{Fore.CYAN}Available commands:")
    print(f"{Fore.YELLOW}  [num1] [num2]  {Fore.WHITE}- Add two numbers (0-1 range works best)")
    print(f"{Fore.YELLOW}  history        {Fore.WHITE}- Show calculation history")
    print(f"{Fore.YELLOW}  help           {Fore.WHITE}- Show this help message")
    print(f"{Fore.YELLOW}  exit           {Fore.WHITE}- Quit the program{Style.RESET_ALL}")

# Display calculation history
def show_history():
    if not history:
        print(f"{Fore.RED}No calculations in history{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.CYAN}Calculation History:{Style.RESET_ALL}")
    for i, entry in enumerate(history, 1):
        timestamp = entry['time'].strftime("%H:%M:%S")
        print(f"{Fore.WHITE}{i}. [{timestamp}] {entry['num1']} + {entry['num2']} = "
              f"{entry['prediction']:.2f} {Fore.MAGENTA}({entry['confidence']:.1f}% confidence){Style.RESET_ALL}")

# Main application
def main():
    model = create_and_train_model()
    
    print(f"\n{Fore.GREEN}AI Calculator Chat Interface{Style.RESET_ALL}")
    print(f"{Fore.BLUE}-------------------------------{Style.RESET_ALL}")
    print_help()
    
    while True:
        try:
            user_input = input(f"\n{Fore.WHITE}You: {Style.RESET_ALL}").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break
                
            if user_input.lower() == 'help':
                print_help()
                continue
                
            if user_input.lower() == 'history':
                show_history()
                continue
                
            # Process numerical input
            numbers = [float(x) for x in user_input.split()]
            
            if len(numbers) != 2:
                print(f"{Fore.RED}Please enter exactly two numbers separated by space{Style.RESET_ALL}")
                continue
                
            num1, num2 = numbers
               
            # Make prediction
            input_array = np.array([[num1, num2]])
            prediction = model.predict(input_array, verbose=0)[0][0]
            actual = num1 + num2
            confidence = calculate_confidence(prediction, actual)
            
            # Add to history
            history.append({
                'time': datetime.now(),
                'num1': num1,
                'num2': num2,
                'prediction': prediction,
                'confidence': confidence
            })
            
            # Format confidence color
            conf_color = Fore.GREEN if confidence > 95 else Fore.YELLOW if confidence > 80 else Fore.RED
            
            print(f"{Fore.GREEN}AI: {num1} + {num2} = {prediction:.2f} "
                  f"{conf_color}({confidence:.1f}% confidence){Style.RESET_ALL}")
            
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter two numbers or a valid command{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()