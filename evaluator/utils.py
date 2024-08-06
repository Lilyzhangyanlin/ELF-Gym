import pandas as pd

def downsample_avito(dataframes, num_rows=100):
    UserID = dataframes['UserInfo'][['UserID']].iloc[:num_rows].reset_index(drop=True)
    dataframes['SearchInfo'] = UserID.merge(dataframes['SearchInfo'], on='UserID', how='left').reset_index(drop=True)
    SearchID = dataframes['SearchInfo'][['SearchID']].reset_index(drop=True)
    dataframes['SearchStream'] = SearchID.merge(dataframes['SearchStream'], on='SearchID', how='left').reset_index(drop=True)
    return dataframes

DOWNSAMPLE_FUNCTIONS = {
    'avito': downsample_avito
}

def downsample(dataframes, column, table_schemas, num_rows=1000):
    table = None
    for table_schema in table_schemas:
        for column_schema in table_schema.columns:
            if column_schema.name == column and column_schema.dtype == 'primary_key':
                table = table_schema.name
    assert table is not None, f"Downsample column should be a primary key."
    new_dataframes = {}
    candidates = dataframes[table][column].unique()[:num_rows]
    for name, df in dataframes.items():
        if column in df.columns:
            df = df[df[column].isin(candidates)]
        new_dataframes[name] = df.reset_index(drop=True)
    return new_dataframes


def downsample_avito_for_generate(dataframes, num_rows=10):
    UserID = dataframes['UserInfo'][['UserID']].iloc[:num_rows].reset_index(drop=True)
    dataframes['SearchInfo'] = UserID.merge(dataframes['SearchInfo'], on='UserID', how='left').reset_index(drop=True)
    SearchID = dataframes['SearchInfo'][['SearchID']].reset_index(drop=True)
    dataframes['SearchStream'] = SearchID.merge(dataframes['SearchStream'], on='SearchID', how='left').reset_index(drop=True)
    return dataframes

def downsample_airbnb_for_generate(dataframes, num_rows=100):
    dataframes['User'] = dataframes['User'].iloc[:num_rows].reset_index(drop=True)
    User_id = dataframes['User'][['id']].iloc[:num_rows].reset_index(drop=True)
    User_id.rename(columns={'id': 'user_id'}, inplace=True)
    dataframes['Session'] = User_id.merge(dataframes['Session'], on='user_id', how='left').reset_index(drop=True)
    return dataframes

def downsample_facebook_for_generate(dataframes, num_rows=100):
    dataframes['Bidders'] = dataframes['Bidders'].iloc[:num_rows].reset_index(drop=True)
    Bidder_id = dataframes['Bidders'][['bidder_id']].iloc[:num_rows].reset_index(drop=True)
    dataframes['Bids'] = Bidder_id.merge(dataframes['Bids'], on='bidder_id', how='left').reset_index(drop=True)
    return dataframes

def downsample_ieee_for_generate(dataframes, num_rows=100):
    dataframes['Transaction'] = dataframes['Transaction'].iloc[:num_rows].reset_index(drop=True)
    Transaction_id = dataframes['Transaction'][['TransactionID']].iloc[:num_rows].reset_index(drop=True)
    dataframes['Train_identity'] = Transaction_id.merge(dataframes['Train_identity'], on='TransactionID', how='left').reset_index(drop=True)
    return dataframes

def downsample_instacert_for_generate(dataframes, num_rows=10):
    User_id = pd.DataFrame({'user_id': dataframes['Orders']['user_id'].unique()[:num_rows]})
    dataframes['Orders'] = User_id.merge(dataframes['Orders'], on='user_id', how='left').reset_index(drop=True)
    Order_id = dataframes['Orders'][['order_id']].reset_index(drop=True)
    dataframes['Order_products__train'] = Order_id.merge(dataframes['Order_products__train'], on='order_id', how='left').reset_index(drop=True)
    dataframes['Order_products__prior'] = Order_id.merge(dataframes['Order_products__prior'], on='order_id', how='left').reset_index(drop=True)
    return dataframes

def downsample_talkingdata_for_generate(dataframes, num_rows=100):
    dataframes['Gender_age'] = dataframes['Gender_age'].iloc[:num_rows].reset_index(drop=True)
    Device_id = dataframes['Gender_age'][['device_id']].iloc[:num_rows].reset_index(drop=True)
    dataframes['Brand'] = Device_id.merge(dataframes['Brand'], on='device_id', how='left').reset_index(drop=True)
    dataframes['Events'] = Device_id.merge(dataframes['Events'], on='device_id', how='left').reset_index(drop=True)
    return dataframes

DOWNSAMPLE_FUNCTIONS_FOR_GENERATE = {
    'avito': downsample_avito_for_generate,
    'airbnb': downsample_airbnb_for_generate,
    'facebook': downsample_facebook_for_generate,
    'ieee': downsample_ieee_for_generate,
    'instacart': downsample_instacert_for_generate,
    'talkingdata': downsample_talkingdata_for_generate
}


