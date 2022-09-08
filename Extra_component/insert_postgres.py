import psycopg2
import pandas as pd


def postgres(data, username, password, port, database, path, table_name, attr_list, prime_key_attr=None):
    """
    Insert the table in POSTGRES.
    The database must already exist!
    """

    pdata = []
    for d in data:
        d2 = {}
        for key in d.keys():
            d2[f'"{key}"'] = d[key]
        pdata.append(d2)

    pattr_list = {}
    for key in attr_list.keys():
        pattr_list[f'"{key}"'] = attr_list[key]

    dff = pd.DataFrame(pdata)
    csv_filepath = path + table_name + '.csv'

    dff.to_csv(csv_filepath, index=False)
    conn_string = f'postgresql://{username}:{password}@localhost:{port}/{database}'
    dbSession = psycopg2.connect(conn_string)

    # Open a database cursor
    dbCursor = dbSession.cursor();
    sqlDrop = f"drop table IF EXISTS {table_name};";
    dbCursor.execute(sqlDrop);

    # SQL statement to create a table
    attr_str = ''
    create_str = f'CREATE TABLE {table_name}('
    for d in pattr_list:
        iskey = ''
        if d == prime_key_attr:
            iskey = ' primary key'
        create_str = create_str + d + ' ' + pattr_list[d] + iskey + ', '
        attr_str += d + ','
    create_str = create_str[:-2] + ')'
    sqlCreateTable = create_str;
    dbCursor.execute(sqlCreateTable);

    sqlImportCsv = f'''COPY {table_name}({attr_str[:-1]})
            FROM '{csv_filepath}'
            DELIMITER ','
            CSV HEADER;'''

    dbCursor.execute(sqlImportCsv);
    dbSession.commit();

    # Close the session and free up the resources
    dbSession.close();
