import os


def check_directory_access(directory_path):
    """
    Checks if the script has access to the specified directory.

    Args:
        directory_path (str): The path to the directory to check.

    Returns:
        bool: True if the directory is accessible, False otherwise.
    """
    try:
        os.listdir(directory_path)  # Use os.listdir to check for access
        return True
    except OSError:
        return False

def main():
    """
    Main function to run the directory access check and print the result.
    """
    directory_to_check = "F:\\data_9"
    
    if check_directory_access(directory_to_check):
        print(f"The directory '{directory_to_check}' is accessible.")
    else:
        print(f"The directory '{directory_to_check}' is NOT accessible.")

if __name__ == "__main__":
    main()
