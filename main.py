import datetime as dt
import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import requests

warnings.filterwarnings('ignore')

DATASET_URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
DATASET_FILENAME = 'owid-covid-data.csv'
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def save_plot(filename, plot):
    fig = plot.get_figure()
    if not os.path.exists("figures"):
        os.mkdir("figures")
    fig.savefig(os.path.join("figures", filename))


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
        min_value = data.loc[data[column].idxmin()]
        max_value = data.loc[data[column].idxmax()]
        print(
            f'Values of column {column}:\n'
            f' Min: {min_value[column]}, Country: {min_value["location"]}\n'
            f' Max: {max_value[column]}, Country: {max_value["location"]}\n'
            f' Mean: {numeric_columns.iloc[:, index].mean()}\n'
            f' Median: {numeric_columns.iloc[:, index].median()}\n'
        )


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
    plot = data.plot.bar(x='date', figsize=(100, 10))
    save_plot("Q2_daily_cases.png", plot)


def draw_monthly_plot(data, country='iran'):
    data = data.query(f'location == "{country.title()}"')
    data['date'] = pd.to_datetime(data['date'])
    data = pd.DataFrame(
        {
            'new_cases': data['new_cases'].groupby(data['date'].dt.to_period('M')).sum(),
            'new_deaths': data['new_deaths'].groupby(data['date'].dt.to_period('M')).sum(),
        }
    )
    plot = data.plot.bar(figsize=(10, 10), rot=0)
    save_plot("Q2_monthly_cases.png", plot)


def draw_weekly_plot(data, country='iran'):
    data = data.query(f'location == "{country.title()}"')
    data = data[['date', 'new_cases', 'new_deaths']]
    data['date'] = pd.to_datetime(data['date'])
    data = data.groupby([pd.Grouper(key='date', freq='W-SUN')])['new_cases', 'new_deaths'] \
        .sum() \
        .reset_index() \
        .sort_values('date')
    data['date'] = data['date'].dt.date
    plot = data.plot.bar(x='date', figsize=(10, 10))
    save_plot("Q2_weekly_cases.png", plot)


def draw_box_whisker_by_countries(data):
    data = data.query('iso_code == "IRN" | '
                      'iso_code == "AFG" | '
                      'iso_code == "IRQ" | '
                      'iso_code == "GBR" | '
                      'iso_code == "ITA"')
    data = data[['location', 'new_cases']]
    plot = data.boxplot(by='location')
    save_plot("Q4_box_whisker_(iran, iraq, afghanistan, uk, italy).png", plot)


def calculate_values(data):
    data = data.query('iso_code == "IRN"')
    q1 = data.new_cases.quantile(.25)
    q3 = data.new_cases.quantile(.75)
    iqr = q3 - q1
    above_iqr = q3 + 1.5 * iqr
    below_iqr = q1 - 1.5 * iqr
    upper_whisker = data.query(f'new_cases <= {above_iqr}').new_cases.max()
    lower_whisker = data.query(f'new_cases >= {below_iqr}').new_cases.min()
    print(
        f"Q values of new cases in iran:\n Q1: {q1}\n Q3: {q3}\n IQR: {iqr}\n Upper Whisker: {upper_whisker}\n "
        f"Lower Whisker: {lower_whisker}\n"
    )

    outliers = data.query(f'new_cases > {upper_whisker}')\
        .sort_values('new_cases', ascending=False)[['date', 'new_cases']].head(10)
    print('10 Outlines from new cases for iran')
    print(outliers)


def draw_iran_box_whisker(data):
    data = data.query('iso_code == "IRN"')
    data = data[['date', 'new_cases']]
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.date
    data['week_date'] = data.apply(lambda row: row['date'] - dt.timedelta(days=row['date'].weekday()), axis=1)
    data = data[['week_date', 'new_cases']]
    plot = data.boxplot(by='week_date', figsize=(50, 10))
    save_plot("bonus_question_1_iran_new_cases_box_whisker.jpg", plot)


def find_missing_values(data):
    data = data.query('iso_code == "FRA"')
    print(f'number of missing values for france new_cases {data["new_cases"].isnull().sum()}')
    missing_values = data.query('new_cases < 0')
    print(f'number of wrong values for france new_cases: {missing_values["new_cases"].count()}')
    for index, row in missing_values.iterrows():
        mean = data.query(f'index == {index + 1} | index == {index - 1}')['new_cases'].mean()
        print(f'Suggested value for column with index {index} is {mean}')
    pass


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
    draw_box_whisker_by_countries(dataset)
    calculate_values(dataset)
    draw_iran_box_whisker(dataset)
    find_missing_values(dataset)
    plt.show()
