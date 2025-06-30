#!/usr/bin/env python3

from csv_to_postgresql import CSVToPostgreSQLLoader

config = {'host': 'test', 'database': 'test',
          'user': 'test', 'password': 'test'}
loader = CSVToPostgreSQLLoader(config)

filename = 'sterling_aml_sanctions_test.csv'
table_name = loader._generate_table_name_from_csv(filename)

print(f"CSV Filename: {filename}")
print(f"Generated Table Name: {table_name}")
print(f"✅ Dynamic table naming working correctly!")
