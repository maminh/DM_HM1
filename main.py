import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import requests

DATASET_URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
DATASET_FILENAME = 'owid-covid-data.csv'
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def get_dataset():
    if os.path.exists(DATASET_FILENAME):
        print('dataset file exists')
        return pd.read_csv(DATASET_FILENAME)
    print('dataset file is not found, retrieving the dataset...')
    try:
        req = requests.get(DATASET_URL)
    except requests.exceptions.RequestException as exc:
        print(f'unable to retrieve dataset, try again or put the dataset in this directory. Exception: {exc}')
        return
    print('dataset file successfully retrieved')
    with open(DATASET_FILENAME, 'w') as dataset_file:
        dataset_file.write(req.text)
    return pd.read_csv(DATASET_FILENAME)


def count_nan_values(data):
    for index, column in enumerate(data.columns):
        print(f'Number of NaN values in column "{column}": {data.iloc[:, index].isnull().sum()}\n')


def get_columns_values(data):
    numeric_columns = data.select_dtypes(include=NUMERICS)
    for index, column in enumerate(numeric_columns.columns):
        print(f'Values of column {column}:\n '
              f'Min: {numeric_columns.iloc[:, index].min()}\n '
              f'Max: {numeric_columns.iloc[:, index].max()}\n '
              f'Mean: {numeric_columns.iloc[:, index].mean()}\n '
              f'Median: {numeric_columns.iloc[:, index].median()}\n')


def get_missing_values(data):
    chosen_columns = [
        'total_cases',
        'new_cases',
        'total_deaths',
        'new_deaths',
        'total_cases_per_million',
        'new_cases_per_million',
        'total_deaths_per_million',
        'new_deaths_per_million',
        'total_tests',
        'new_tests'
    ]

    for column in chosen_columns:
        query_count = data.query(f'{column} < 0').count()[column]
        print(f'number of invalid values (negative values) in column {column}: {query_count}\n')


def draw_daily_plot(data, country='iran'):
    data = data.query(f'location == "{country.title()}"')
    data = data[['date', 'new_cases', 'new_deaths']]
    data.plot.bar(x='date', figsize=(255, 10))
    plt.show()


def draw_monthly_plot(data, country='iran'):
    data = data.query(f'location == "{country.title()}"')
    data['date'] = pd.to_datetime(data['date'])
    data = pd.DataFrame(
        {
            'new_cases': data['new_cases'].groupby(data['date'].dt.to_period('M')).sum(),
            'new_deaths': data['new_deaths'].groupby(data['date'].dt.to_period('M')).sum(),
        }
    )
    data.plot.bar(figsize=(10, 10), rot=0)
    plt.show()


def draw_weekly_plot(data, country='iran'):
    data = data.query(f'location == "{country.title()}"')
    data = data[['date', 'new_cases', 'new_deaths']]
    data['date'] = pd.to_datetime(data['date'])
    data = data.groupby([pd.Grouper(key='date', freq='W-SUN')])['new_cases', 'new_deaths'] \
        .sum() \
        .reset_index() \
        .sort_values('date')
    data['date'] = data['date'].dt.date
    data.plot.bar(x='date', figsize=(10, 10))
    plt.show()


if __name__ == '__main__':
    dataset = get_dataset()
    if dataset is None:
        sys.exit(-1)
    count_nan_values(dataset)
    get_columns_values(dataset)
    get_missing_values(dataset)
    draw_daily_plot(dataset)
    draw_weekly_plot(dataset)
    draw_monthly_plot(dataset)
