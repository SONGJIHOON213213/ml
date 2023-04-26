import pandas as pd

# PM, AWS 데이터 로드
pm_data = pd.read_csv('c:/study/_data/ggamji/META/pmmap.csv')
aws_data = pd.read_csv('c:/study/_data/ggamji/META/awsmap.csv')

# TRAIN 폴더 안의 pm 데이터 불러오기
path = 'c:/study/_data/ggamji/META/'
path2 = 'c:/study/_data/ggamji/Train/'

train = pd.DataFrame()
for i in range(1, 11):
    file_name = f'train_{i}.csv'
    df = pd.read_csv(path2 + file_name, parse_dates=['일시'])
    train = pd.concat([train, df])