import pandas as pd
import kmeans_file
import os

def main(folder):
    file_path = 'input' + '/' + folder
    files = os.listdir(file_path)

    while True:
        print('\n Choose a file: ')
        for f in range(len(files)):
            print(f"{f}: {files[f]}")
        try:
            file_num = int(input())
            if file_num < 0 or file_num >= len(files):
                print("Invalid choice. Please choose a valid number.")
            else:
                break
        except ValueError:
            print('Not a valid entry.')
    df = pd.read_excel(file_path + '/' + files[file_num])

    while True:
        print('\n Choose a column: ')
        for c in range(len(df.columns)):
            print(f"{c}: {df.columns[c]}")
        try:
            col_num = int(input())
            if col_num < 0 or col_num >= len(df.columns):
                print("Invalid choice. Please choose a valid number.")
            else:
                break
        except ValueError:
            print('Not a valid entry.')

    df = df[df['School'] == 'Lord Tweedsmuir']
    question = df.columns[col_num]
    responses = df[question].fillna('')
    listed_response = kmeans_file.main(responses)
    return listed_response, os.path.splitext(files[file_num])[0]
    