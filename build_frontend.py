import os
import subprocess
import sys

def main():
    """Build the React frontend for the dashboard"""
    print("Building frontend for Federated Ensemble Learning Dashboard...")
    
    # Ensure the frontend directory exists
    if not os.path.exists('frontend'):
        print("Error: frontend directory not found.")
        return False
    
    # Check if node is installed
    try:
        subprocess.run(['node', '--version'], check=True, stdout=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: Node.js is not installed or not in PATH.")
        print("Please install Node.js from https://nodejs.org/")
        return False
    
    # Navigate to frontend directory
    os.chdir('frontend')
    
    # Install dependencies
    print("Installing dependencies...")
    try:
        subprocess.run(['npm', 'install'], check=True)
    except subprocess.SubprocessError:
        print("Error installing dependencies. Please check npm is installed correctly.")
        os.chdir('..')
        return False
    
    # Build the frontend
    print("Building frontend...")
    try:
        subprocess.run(['npm', 'run', 'build'], check=True)
    except subprocess.SubprocessError:
        print("Error building frontend.")
        os.chdir('..')
        return False
    
    # Return to the root directory
    os.chdir('..')
    
    print("Frontend built successfully!")
    print("You can now run the dashboard with: python dashboard.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 