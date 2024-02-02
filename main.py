import subprocess

def run_model_script():
    subprocess.run(["python", "model.py"])

def main():
    # Your main code logic goes here
    print("Running main.py")
    # Call the function to run model.py
    run_model_script()

if __name__ == '__main__':
    main()

