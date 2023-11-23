import os
import pandas as pd


def make_annotations_file(dataset_path: str, type: str) -> None:

    '''
    Generates an annotation file based on elements in the specified folder.

    Parameters:
    - dataset_path (str): Path to the folder containing the data.
    - type (str): Type of annotation to include in the file name.

    Returns:
    None

    Reads the list of elements in the dataset_path folder, sorts them, and creates a DataFrame
    with a 'labels' column, where 1 indicates that the element name contains 'opened', and 0 otherwise.
    Saves the created DataFrame to a CSV file in the './Annotations/' folder
    with the name 'annotation_file_{type}.csv' without adding an index column.

    Example:
    make_annotations_file('/path/to/folder', 'annotation_type')
    '''

    list_of_elements = sorted(os.listdir(dataset_path))

    annotation = pd.DataFrame()
    annotation['labels'] = [1 if 'opened' in element else 0 for element in list_of_elements]
    
    annotation.to_csv(path_or_buf=f'./Annotations/annotation_file_{type}.csv', sep=',', index=False)