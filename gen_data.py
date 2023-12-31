import mysql.connector
import numpy as np
import datetime
import random


last_fetched_timestamp = None

def get_latest_data_from_database(host, user, password, database):
    """
    从数据库获取最新的人流量数据。
    """
    global last_fetched_timestamp

    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = cnx.cursor()

    query = "SELECT id, timestamp, flow_count FROM flow_of_people WHERE timestamp > %s ORDER BY timestamp"
    cursor.execute(query, (last_fetched_timestamp,))
    records = cursor.fetchall()
    
    if records:
        data = np.array([[record[0], record[1].year, record[1].month, record[1].day, record[1].hour, record[1].minute, record[1].second, record[2]] for record in records])
        last_fetched_timestamp = records[-1][0]
    else:
        data = None

    cursor.close()
    cnx.close()

    return data


def delete_anomaly_from_database(host, user, password, database, anomaly_id):
    """
    从数据库删除异常值。
    """
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = cnx.cursor()

    cursor.execute("DELETE FROM pedestrian_flow WHERE id = %s", (anomaly_id,))
    cnx.commit()

    cursor.close()
    cnx.close()


def get_data_from_database(host, user, password, database):
    """
    从数据库获取人流量数据。

    参数:
    - host: 数据库主机地址
    - user: 数据库用户名
    - password: 数据库密码
    - database: 使用的数据库名称
    
    返回值:
    一个二维数组，其中每一行包含年、月、日、小时、分钟、秒和对应的人流量。
    """

    #获取数据库中的所有数据并设置last_fetched_timestamp为最新的时间戳。
    global last_fetched_timestamp

    # 连接数据库
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = cnx.cursor()

    # 查询pedestrian_flow表中的所有数据
    cursor.execute("SELECT id, timestamp, count FROM pedestrian_flow ORDER BY timestamp")
    records = cursor.fetchall()
    last_fetched_timestamp = records[-1][0]

    data = np.array([[record[0], record[1].year, record[1].month, record[1].day, record[1].hour, record[1].minute, record[1].second, record[2]] for record in records])

    # 在完成后关闭cursor和连接
    cursor.close()
    cnx.close()
    
    return data


# 生成模拟的人流量数据的函数
def generate_flow_data(start_date, days):
    def get_flow(current_hour):
        if 0 <= current_hour < 6:
            return random.randint(0, 5)
        elif 6 <= current_hour < 9:
            return random.randint(50, 100)
        elif 9 <= current_hour < 12:
            return random.randint(30, 70)
        elif 12 <= current_hour < 14:
            return random.randint(60, 120)
        elif 14 <= current_hour < 18:
            return random.randint(20, 60)
        elif 18 <= current_hour < 21:
            return random.randint(50, 100)
        else:
            return random.randint(0, 20)
    
    data = []
    for day in range(days):
        current_time = start_date + datetime.timedelta(days=day)
        end_time = current_time + datetime.timedelta(days=1)
        while current_time < end_time:
            flow = get_flow(current_time.hour)
            data.append((current_time, flow))
            current_time += datetime.timedelta(seconds=10)
    return data


def main():
    # 连接数据库
    cnx = mysql.connector.connect(
        host="localhost",
        user="mysql",
        password="123456",
        database="flow_of_people"
    )
    cursor = cnx.cursor()
    
    # 检查数据库中是否存在pedestrian_flow表
    cursor.execute("SHOW TABLES LIKE 'pedestrian_flow'")
    result = cursor.fetchone()
    
    # 如果表不存在，则创建它
    if not result:
        cursor.execute("""
            CREATE TABLE pedestrian_flow (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                count INT NOT NULL
            )
        """)
    
    # 生成7天的模拟数据
    start_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    flow_data = generate_flow_data(start_date, 7)
    
    # 将数据插入数据库
    for record in flow_data:
        cursor.execute("""
            INSERT INTO pedestrian_flow (timestamp, count)
            VALUES (%s, %s)
        """, record)
    
    # 提交事务
    cnx.commit()
    
    # 查询和显示数据（仅作为示例，按时间戳排序）
    cursor.execute("SELECT * FROM pedestrian_flow ORDER BY timestamp LIMIT 10")  # 显示前10条数据
    print("First 10 records:")
    for row in cursor.fetchall():
        print(row)
    
    # 关闭cursor和连接
    cursor.close()
    cnx.close()
    

def create_planned_events_table(host, user, password, database):
    """ 
    在数据库中创建'planned_events'表，如果它不存在的话。
    """
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = cnx.cursor()

    cursor.execute("SHOW TABLES LIKE 'planned_events'")
    result = cursor.fetchone()

    if not result:
        cursor.execute("""
            CREATE TABLE planned_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event_date DATE NOT NULL,
                start_time TIME NOT NULL,
                end_time TIME NOT NULL,
                expected_count INT NOT NULL
            )
        """)

    cursor.close()
    cnx.close()


def add_planned_event(host, user, password, database, event_date, start_time, end_time, expected_count):
    """ 
    向'planned_events'表中添加一个预定事件。
    """
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = cnx.cursor()

    cursor.execute("""
        INSERT INTO planned_events (event_date, start_time, end_time, expected_count)
        VALUES (%s, %s, %s, %s)
    """, (event_date, start_time, end_time, expected_count))

    cnx.commit()

    cursor.close()
    cnx.close()


def get_planned_event(host, user, password, database, timestamp):
    """ 
    根据给定的时间戳从数据库中获取预定的事件。
    """
    cnx = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = cnx.cursor()

    cursor.execute(
        "SELECT expected_count FROM planned_events \
        WHERE event_date = %s AND start_time <= %s AND end_time >= %s",
        (timestamp.date(), timestamp.time(), timestamp.time())
    )
    record = cursor.fetchone()

    cursor.close()
    cnx.close()

    return record[0] if record else None


if __name__ == "__main__":
    main()