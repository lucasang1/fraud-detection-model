# define cleaning methods
def drop_unused(df):
    return df.drop(columns = ['row_num']) # not sure if this is needed

def split_X_y(df):
    X = df.drop(columns = ['class']) # everything except class
    y = df['class'] # class only
    return X, y 