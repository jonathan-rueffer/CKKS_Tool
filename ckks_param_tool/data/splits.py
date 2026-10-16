from sklearn.model_selection import train_test_split

'''
This file contains the function that splits the dataset for simulating deployment scenarios
'''

# server_size is the fraction of the dataset given to the server for training the model; the
# remaining rows form the client's (unlabelled) data
def client_server_split(X, y, server_size=0.3, random_state=42):
    X_client, X_server, y_client, y_server = train_test_split(
        X, y, test_size=server_size, random_state=random_state
    )
    return X_client, X_server, y_client, y_server
