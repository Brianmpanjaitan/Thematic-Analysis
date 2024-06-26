import contractions
import re
import sql_data

def setupTRAINED(df):
    df_filtered = df
    df_filtered['Sentence'] = df_filtered['Sentence'].apply(preprocessText)
    return df_filtered

def preprocessText(text):
    if text is None:
        return ''
    
    expanded_words = []    
    for word in text.split():
        expanded_words.append(contractions.fix(word)) 
    expanded_text = ' '.join(expanded_words)
    cleaned_text = re.sub(r'[^\w\s]', '', expanded_text).lower()
    return cleaned_text

def main():
    df = sql_data.main()
    df_filtered = setupTRAINED(df)
    return df_filtered