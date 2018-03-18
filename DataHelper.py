# -*- Encoding:UTF-8 -*-

import pandas as pd


class Data:
    def __init__(self, name='ml-1m'):
        self.dataName = name
        self.dataPath = "./data/" + self.dataName + "/"
        # Static Profile
        self.UserInfo = self.getUserInfo()
        self.MovieInfo = self.getMovieInfo()

        self.data = self.getData()

    def getUserInfo(self):
        if self.dataName == "ml-1m":
            userInfoPath = self.dataPath + "users.dat"

            users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
            users = pd.read_table(userInfoPath, sep='::', header=None, names=users_title, engine='python')
            users = users.filter(regex='UserID|Gender|Age|JobID')
            users_orig = users.values

            # 将性别映射到0,1
            gender_map = {'F': 0, 'M': 1}
            users['Gender'] = users['Gender'].map(gender_map)
            # 将年龄组映射到0-6
            age_map = {val: idx for idx, val in enumerate(set(users['Age']))}
            users['Age'] = users['Age'].map(age_map)

            return users

    def getMovieInfo(self):
        if self.dataName == "ml-1m":
            movieInfoPath = self.dataPath + "movies.dat"

            movies_title = ['MovieID', 'Title', 'Genres']
            movies = pd.read_table(movieInfoPath, sep='::', header=None, names=movies_title, engine='python')
            movies = movies.filter(regex='MovieID|Genres')

            #电影类型映射到0-18
            genres_set = set()
            for val in movies['Genres'].str.split('|'):
                genres_set.update(val)
            genres2int = {val: idx for idx, val in enumerate(genres_set)}
            genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}
            movies['Genres'] = movies['Genres'].map(genres_map)

            return movies

    def getData(self):
        if self.dataName == "ml-1m":
            dataPath = self.dataPath + "ratings.dat"

            ratings_title = ['UserID', 'MovieID', 'Rating', 'TimeStamp']
            ratings = pd.read_table(dataPath, sep='::', header=None, names=ratings_title, engine='python')

            data = pd.merge(pd.merge(ratings, self.UserInfo), self.MovieInfo)
            data = data.sort_values(by=['TimeStamp'])

            return data


if __name__ == '__main__':
    data = Data()
    print(data.MovieInfo)


