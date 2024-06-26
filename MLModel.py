import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import date
import contractions
import re
import warnings
import nltk

import setup_train_data
import input_file
import sql_data
warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()
resources = ['stopwords', 'punkt', 'wordnet']
stop_words = set(stopwords.words('english'))
project_name_list = ['TRANSITIONS', 'ELPATS', 'SEL']

def download_if_not_exists(resource_name):
    try:
        nltk.data.find(f'tokenizers/{resource_name}')
        print(f"{resource_name} is already downloaded.")
    except LookupError:
        print(f"Downloading {resource_name}...")
        nltk.download(resource_name, quiet=True)

def decideTheme(user_input, theme_list):
    if user_input >= len(theme_list) - 1:
        text = input('Enter your own theme: \n')   
        return text     
    return theme_list[user_input]

def exportDataToTrainingSet(df, folder, filename):
    print(df)
    satisfied_input = input("Are you satisfied with the resulting themes? (y/N)\n").lower()
    while satisfied_input not in ['y','n']:
        print("Invalid input. Please enter 'y' or 'n'.")
        satisfied_input = input("Are you satisfied with the resulting themes? (y/N)\n").lower()
    if satisfied_input == 'y':
        today = str(date.today())
        #excel_file_path = 'train' + '/' + folder + '/' + f"TRAINED_{filename}_{today}.xlsx"
        #df.to_excel(excel_file_path, index=False)
        sql_data.uploadData(df, f"TRAINED_{filename}_{today}")
        return
    elif satisfied_input == 'n':
        while True:
            try:
                adjust_row = int(input(f"Which row would you like to adjust? (0-{len(df)-1})\n"))
                if 0 <= adjust_row < len(df):
                    text = input("What would you like the new result to be?\n")
                    df.loc[adjust_row, 'Theme'] = text 
                    break
                else:
                    print('Not a valid option')
            except ValueError:
                print('Error: Enter Again')
    exportDataToTrainingSet(df, folder, filename)

def preprocessData(df):
    df.fillna("", inplace=True)
    df_melted = df.melt(var_name='Theme', value_name='Sentence')
    df_melted = df_melted[df_melted['Sentence'] != ""]
    df_melted['Sentence'] = df_melted['Sentence'].apply(preprocessText)
    return df_melted

def preprocessText(text):
    expanded_words = []    
    for word in text.split():
        expanded_words.append(contractions.fix(word)) 
    expanded_text = ' '.join(expanded_words)
    cleaned_text = re.sub(r'[^\w\s]', '', expanded_text).lower()
    return cleaned_text

def buildModel(df, response, auto_theme_input):
    X = df['Sentence']
    y = df['Theme']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train_tfidf, y_train)

    y_test_proba = random_forest.predict_proba(X_test_tfidf)
    
    # Evaluate Thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba[:, 1], pos_label=random_forest.classes_[1])
    f1_scores = np.divide(2 * (precision * recall), (precision + recall), where=(precision + recall) != 0)
    f1_scores = np.where((precision + recall) == 0, 0, f1_scores)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]

    result_list = []
    for text in response:
        preprocessed_text = preprocessText(text)
        text_tfidf = vectorizer.transform([preprocessed_text])
        probabilities = random_forest.predict_proba(text_tfidf)[0]
        
        predicted_themes = [(theme, prob) for theme, prob in zip(random_forest.classes_, probabilities) if prob >= best_threshold]
        predicted_themes.sort(key=lambda x: x[1], reverse=True)
        top_themes = predicted_themes[:2] # Top 2 Themes

        if top_themes:
            if auto_theme_input == 'y':
                theme1, prob = top_themes[0]
                result_list.append({'Sentence': text, 'Theme': theme1})

            elif auto_theme_input == 'n':
                print(f"'{text}':")
                theme_list = [theme for i, (theme, prob) in enumerate(top_themes, start=0)]
                theme_list.append('Neither')
                for i, theme in enumerate(theme_list):
                    print(f"{i}: {theme}, Probability: {top_themes[i][1]:.2f}" if i < len(top_themes) else f"{i}: {theme}")
                while True:
                    try:
                        user_input = int(input(f"\nWhich option accurately represents the response? 0-{len(theme_list) - 1}\n"))
                        if 0 <= user_input < len(theme_list):
                            theme = decideTheme(user_input, theme_list)
                            result_list.append({'Sentence': text, 'Theme': theme})
                            break
                        else:
                            print('Not a valid option')
                    except ValueError:
                        print('Error: Enter Again')
        else:
            print(f"The sentence '{text}' does not belong to any theme with probability above {best_threshold}")
    result_df = pd.DataFrame(result_list)
    return result_df

def main():
    for resource in resources:
        download_if_not_exists(resource)

    while True:
        print("Select the folder that contains your data: \n 0 - Transitions, 1 - ELPATS, 2 - SEL")
        export_df = pd.DataFrame()
        try:
            user_input = int(input())
            if 0 <= user_input < len(project_name_list):
                df = setup_train_data.main()
                responses, filename = input_file.main(project_name_list[user_input])

                auto_theme_input = input("Would you like to automatically theme all responses? (y/N)\n").lower()
                while auto_theme_input not in ['y','n']:
                    print("Invalid input. Please enter 'y' or 'n'.")
                    auto_theme_input = input("Would you like to automatically theme all responses? (y/N)\n").lower()

                for response in responses:
                    result_df = buildModel(df, response, auto_theme_input)
                    export_df = pd.concat([export_df, result_df], ignore_index=True)

                exportDataToTrainingSet(export_df, project_name_list[user_input], filename)
            else:
                print('Out of Bounds')

        except ValueError:
            print("Error: Program Couldn't Execute")

if __name__ == "__main__":
    main()
    