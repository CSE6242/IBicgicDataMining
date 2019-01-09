import pymysql

DATABASE = "testyuhao"
USER = "testyuhao"
PASSWORD = "testyuhao"
HOST = "testyuhao.cebyxwdtjyxd.us-east-1.rds.amazonaws.com"
PORT = 3306


conn = pymysql.connect(db=DATABASE, user=USER, passwd=PASSWORD,
                           host=HOST, port=PORT)

print("OK!")