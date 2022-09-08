
if __name__ == "__main__":
	from auxiliary_defs import primaryKey,det_dialect, datatype_identify,parse_pytheas_annotation
	import pandas as pd
	import argparse
	import sys
	import dashboard_csv
	from clustering import table_clustering

	parser = argparse.ArgumentParser()    
	parser.add_argument("-a", "--annotations", default = '..\Pytheas\src\pytheas\inferred_annotation.json')#, description="Filepath to pre-trained rule weights")
	parser.add_argument("-c", "--csvpath", default = None)#, description="Filepath to CSV file over which to infer annotations")
	#parser.add_argument("-o", "--output_file_path", default = None')
	#parser.add_argument("-d", "--database_system",) # default = 'None')

	args = parser.parse_args(sys.argv[1:])
	csv_file = args.csvpath
	annot_file = args.annotations
	#output_file = args.output_file
	#database_system = args.database_system

	csv_file_name = csv_file[csv_file.rfind('/') + 1:-4]
	df_csv = pd.read_csv(csv_file, delimiter=det_dialect(csv_file), header=None)

	# Parse Pytheas output in a dictionary #
	tables_dict = parse_pytheas_annotation(annot_file, df_csv, csv_file_name)

	final_tables = {}
	for table in tables_dict:
		tab = {}
		dict_data, dict_attr, inconsistent_cols = datatype_identify(tables_dict[table]['table_data'])
		tab['inconsistent_cols'] = inconsistent_cols
		tab['attr_names'] = dict_attr
		tab['table_data'] = pd.DataFrame.from_dict({your_key: dict_data[your_key] for your_key in tab['attr_names'].keys()})
		tab['metadata'] = tables_dict[table]['df_metadata']
		tab['footnotes'] = tables_dict[table]['df_foot']
		tab['confidence'] = tables_dict[table]['conf']
		tab['clustering_results'] = table_clustering(tab['table_data'], tab['attr_names'], tab['inconsistent_cols'])
		final_tables[table] = tab

	# TODO : check if numeric column has categorical role
	# TODO: pass infos-logs about inconsistent cols and clustering

	# Create a dashboard using Plotly package #
	# Follow the link in where the Dash is running (command prompt)
	dashboard_csv.create_dash(csv_file_name+'.csv',final_tables)


'''
	db_systems = ['postgres', 'mongo']
	df_set=[]
	if database_system in db_systems :
		for table_name in final_tables:
			data_folder_postgres = 'C:/Program Files/PostgreSQL/13/'
			csvtable_path = data_folder_postgres + table_name + '.csv'

			prime_key_attr = primaryKey(final_tables[table_name]['table_data'])
			print ('prime: ' ,prime_key_attr)
			df_set.append(final_tables[table_name]['table_data'])
			final_tables[table_name]['table_data'].to_csv(csvtable_path, encoding='utf-8', index=False)
			#postgres(csvtable_path,final_tables[table_name]['attr_names'],prime_key_attr)

	#ls = lens.summarise(csv_table)
	#dashboard.create_dash(csvtable_path)
	#dash_example1.create_dash(csv_file_name+'.csv',final_tables)
	#fetch_data_from_postgres_Dash.create_dash(csv_file_name+'.csv',final_tables)
'''


