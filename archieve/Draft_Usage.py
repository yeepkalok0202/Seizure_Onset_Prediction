import pickle


def read_pkl_file(file_path):
    """
    Reads data from a .pkl file.

    Args:
        file_path (str): The path to the .pkl file.

    Returns:
        The data loaded from the .pkl file, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Successfully read data from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except EOFError:
        print(f"Error: End of file error. The file '{file_path}' might be empty or corrupted.")
        return None
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle the data. The file '{file_path}' may be corrupted or not a valid pickle file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{file_path}': {e}")
        return None

if __name__ == "__main__":
    # Replace 'your_file.pkl' with the actual path to your .pkl file
    file_to_read = 'processed_data_pytorch_adaptive_pre_post_buffer_lovo_personalization_v2_enhanced/cached_processed_data/processed_patient_data_n10_piw_60_pieb_180_pib_180_sf_1.pkl' 
    print(f"\nAttempting to read: {file_to_read}")
    loaded_data = read_pkl_file(file_to_read)

    if loaded_data is not None:
        print("\n--- Data Content ---")
        print(loaded_data)
        print("--------------------")

        # You can now work with the loaded_data
        # For example, if it's a dictionary:
        if isinstance(loaded_data, dict):
            print("\nAccessing some elements (if it's a dictionary like the dummy data):")
            print(f"Name: {loaded_data.get('name')}")
            print(f"Scores: {loaded_data.get('scores')}")