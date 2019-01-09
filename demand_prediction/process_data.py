from util.DbOprartions import *
import requests



def addNearby():
    sql1 = 'SELECT id, latitude, longitude FROM stations'
    stations = getAllResults(sql1)
    for station in stations:
        ##sql2 = 'SELECT COUNT(*) from subway_stations WHERE ST_DISTANCE_SPHERE(geom, ST_SetSRID(ST_MakePoint(' + str(station[2]) + \
        ##       ',' + str(station[1]) + '), 4326)) < 2000'
        sql2 = 'SELECT restaurant from nearby'
        restaurant_num = getOneResult(sql2)[0]

        ##sql3 = 'SELECT COUNT(*) from bus_stops WHERE ST_DISTANCE_SPHERE(geom, ST_SetSRID(ST_MakePoint(' + str(station[2]) + \
        ##       ',' + str(station[1]) + '), 4326)) < 2000'
        sql3 = 'SELECT bus from nearby'
        bus_num = getOneResult(sql3)[0]

        sql4 = 'SELECT subway from nearby'
        subway_num = getOneResult(sql4)[0]
        
        sql5 = 'UPDATE stations SET restaurant = ' + str(subway_num) + ',' + 'bus = ' + str(bus_num) + 'subway = ' + str(subway_num) + ' WHERE id = ' + str(station[0])
        
        execute(sql5)
'''
def addStationCapacity():
    response = requests.get('https://gbfs.citibikenyc.com/gbfs/en/station_information.json')
    station_infos = response.json()['data']['stations']
    for station_info in station_infos:
        station_id = station_info['station_id']
        station_capacity = station_info['capacity']
        sql = 'UPDATE stations SET capacity = ' + str(station_capacity) + ' WHERE id = ' + station_id
        execute(sql)

'''
'''
def addWeatherToTrips():
    sql1 = 'SELECT * FROM central_park_weather'
    weathers = getAllResults(sql1)
    for weather in weathers:
        sql2 = 'UPDATE daily_trips_and_weather SET max_temperature = ' + str(weather[4]) + ', min_temperature = ' + \
               str(weather[5]) + ', average_wind_speed = ' + str(weather[6]) + ', precipitation = ' + \
               str(weather[1]) + ', snow_fall = ' + str(weather[3]) + ', snow_depth = ' + str(weather[2]) + \
               ' WHERE date= \'' + str(weather[0]) + '\''
        execute(sql2)
'''

def addWeatherToTrips():
    sql1 = 'SELECT * FROM central_park_weather'
    weathers = getAllResults(sql1)
    for weather in weathers:
        sql2 = 'UPDATE hourly_trips_and_weather SET weather = '+ str(weather[1]) + ', temperature = ' + str(weather[5]) + ', humidity = ' + \
               str(weather[6]) + ', pressure = ' + str(weather[7]) + ', windSpeed = ' + \
               str(weather[8]) + ', windBearing = ' + str(weather[9]) + ', visibility = ' + str(weather[10]) + \
               ' WHERE w_date= \'' + str(weather[0]) + '\''
        execute(sql2)

'''
def addNYCT():
    sql = 'UPDATE stations SET nyct2010_gid = n.gid, boroname = n.boroname, ntacode = n.ntacode, ntaname = n.ntaname ' \
          'FROM nyct2010 n WHERE stations.nyct2010_gid IS NULL AND ST_Within(stations.geom, n.geom)'

    execute(sql)

def addTaxiZone():
    sql = 'UPDATE stations SET taxi_zone_gid = z.gid, taxi_zone_name = z.zone FROM taxi_zones z ' \
          'WHERE stations.taxi_zone_gid IS NULL AND ST_Within(stations.geom, z.geom)'

    execute(sql)
'''

'''
def addALLData():
    sql1 = 'INSERT INTO demand_prediction_data SELECT month, dow, weekday, holiday, max_temperature, min_temperature, ' \
           'precipitation, average_wind_speed, snow_fall, snow_depth, capacity, nyct2010_gid, taxi_zone_gid, subway, ' \
           'bus, each_station_daily_trips.trips FROM stations JOIN each_station_daily_trips ' \
           'ON stations.id = each_station_daily_trips.id JOIN daily_trips_and_weather ' \
           'ON each_station_daily_trips.date = daily_trips_and_weather.date ' \
           'WHERE capacity IS NOT NULL AND nyct2010_gid IS NOT NULL'
    execute(sql1)
'''

def addALLData():
    sql1 = 'INSERT INTO demand_prediction_data SELECT month, dow, weekday, holiday, weather, temperature, ' \
           'humidity, pressure, windSpeed, windBearing,visibility, restaurant, bus, subway, ' \
           'each_station_hourly_trips.trips FROM stations JOIN each_station_hourly_trips ' \
           'ON stations.id = each_station_hourly_trips.id JOIN hourly_trips_and_weather ' \
           'ON each_station_hourly_trips.starttime = hourly_trips_and_weather.w_date ' 
    execute(sql1)



def main():
    addNearby()
    addStationCapacity()
    addWeatherToTrips()
    ##addNYCT()
    ##addTaxiZone()
    addALLData()

if __name__ == "__main__":
    main()
