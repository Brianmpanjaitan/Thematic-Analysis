import pandas as pd
import pyodbc
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

DRIVER_NAME = 'ODBC Driver 17 for SQL Server'
SERVER_NAME = 'PRD-AS24'
DATABASE_NAME = 'Research'
connectionString = (
    f"DRIVER={{{DRIVER_NAME}}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DATABASE_NAME};"
    f"Trusted_Connection=yes;"
)

try:
    conn = pyodbc.connect(connectionString)
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit()

cursor = conn.cursor()
query = """
SELECT TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'MLModel'
"""
cursor.execute(query)
tables = cursor.fetchall()

def getValidSelection(table_names):
    while True:
        try:
            selected_indices = input("Enter the numbers of the tables you want to select, separated by commas: ")
            selected_indices = [int(i) for i in selected_indices.split(',')]
            # Remove duplicates by converting to set and back to list
            selected_indices = list(set(selected_indices))
            if all(0 <= i < len(table_names) for i in selected_indices):
                return selected_indices
            else:
                print(f"Error: Please enter numbers between 0 and {len(table_names) - 1}.")
        except ValueError:
            print("Error: Please enter valid integers separated by commas.")

def uploadData(df, table_name):
    cursor = conn.cursor()

    cols = ", ".join([f"[{col}]" if "-" in col else col + " NVARCHAR(MAX)" for col in df.columns])
    create_table_query = f"CREATE TABLE MLModel.[{table_name}] ({cols});"
    try:
        cursor.execute(create_table_query)
        conn.commit()
    except Exception as e:
        print(f"Error creating table: {e}")
        return

    for index, row in df.iterrows():
        placeholders = ", ".join(["?" for _ in row])
        insert_query = f"INSERT INTO MLModel.[{table_name}] VALUES ({placeholders})"
        try:
            cursor.execute(insert_query, tuple(row))
        except Exception as e:
            print(f"Error inserting row: {e}")

    conn.commit()
    cursor.close()
    print(f"DataFrame successfully uploaded to table {table_name}")

def main():
    print("Select the Tables you would like to Train the Model on:")
    table_names = [table.TABLE_NAME for table in tables]
    for idx, table_name in enumerate(table_names):
        print(f"{idx}: {table_name}")

    selected_indices = getValidSelection(table_names)
    selected_tables = [table_names[i] for i in selected_indices]

    data_frames = []
    for table_name in selected_tables:
        query = f"SELECT * FROM MLModel.{table_name}"
        data_frames.append(pd.read_sql(query, conn))

    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df