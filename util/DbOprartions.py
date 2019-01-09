import pymysql

def getConn():
    DATABASE = "testyuhao"
    USER = "testyuhao"
    PASSWORD = "testyuhao"
    HOST = "testyuhao.cebyxwdtjyxd.us-east-1.rds.amazonaws.com"
    PORT = 3306

    try:
        conn = pymysql.connect(db=DATABASE, user=USER, passwd=PASSWORD,
                                host=HOST, port=PORT)
    except pymysql.OperationalError:
        print('Fail connecting database!')
    else:
        return conn


def getAllResults(sql):
    conn = getConn()
    cur = conn.cursor()
    cur.execute(sql)
    results = cur.fetchall()
    conn.commit()
    conn.close()
    return results


def getOneResult(sql):
    conn = getConn()
    cur = conn.cursor()
    cur.execute(sql)
    result = cur.fetchone()
    conn.commit()
    conn.close()
    return result