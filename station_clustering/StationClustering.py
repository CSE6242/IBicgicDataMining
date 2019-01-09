from util.DbOprartions import *
from kmeans import K_Means
from numpy import *


def main():

    # 获取数据
    sql = 'SELECT * FROM station_hour_trips ORDER BY stationId;'
    station_hourly_trips = getAllResults(sql)
    data = zeros((len(station_hourly_trips), 24))
    for i in range(len(station_hourly_trips)):
        data[i] = array(station_hourly_trips[i][1:25])

    # K-means聚类
    k_means = K_Means()
    k_means.fit(data)

    # 记录结果
    print(len(k_means.clf_[0]), len(k_means.clf_[1]), len(k_means.clf_[2]), len(k_means.clf_[3]))

    for station_trip in station_hour_trips:
        stationType = k_means.predict(array(station_trip[1:25]))
        sql = 'UPDATE station_hour_trips SET stationType = ' + str(stationType) + ' WHERE stationId = ' + str(station_trip[0];)
        execute(sql)

if __name__ == "__main__":
    main()


