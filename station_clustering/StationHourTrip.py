import pymysql
from util.DbOprartions import *

def main():
    DATABASE = "testyuhao"
    USER = "testyuhao"
    PASSWORD = "testyuhao"
    HOST = "testyuhao.cebyxwdtjyxd.us-east-1.rds.amazonaws.com"
    PORT = 3306

    conn = pymysql.connect(db=DATABASE, user=USER, passwd=PASSWORD,
                           host=HOST, port=PORT)
    i = 0
    try:
        with conn.cursor() as cursor:
            sql2 = 'SELECT startStationId, date_format(startTime,\'%H\') as hour, count(*) as count FROM Bike GROUP BY startStationId, date_format(startTime,\'%H\') ORDER BY startStationId, date_format(startTime,\'%H\');'
            station_trips = getAllResults(sql2)
            ids = set()
            for data in station_trips:
                ids.add(data[0])

            for sid in ids:
                sql5 = "INSERT INTO station_hour_trips(stationId) VALUES(" + str(sid) + ");"
                cursor.execute(sql5)
                conn.commit()



            for data in station_trips:
                stationId = data[0]
                hour = data[1]
                count = data[2]
                sql3 = 'UPDATE station_hour_trips SET hour' + hour + ' = ' + str(count) + ' WHERE stationId =' + str(stationId) + ';'
                cursor.execute(sql3)
                conn.commit()
                print(i)
                i += 1
    # If the usage is 0, update it to 0/
            for i in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                      '16', '17', '18', '19', '20', '21', '22', '23']:
                sql4 = 'UPDATE station_hour_trips SET hour' + str(i) + ' = 0 WHERE hour' + str(i) + ' IS NULL'
                cursor.execute(sql4)
                conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()