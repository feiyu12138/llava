import json

def filter_json_by_id(data, list_ids):
  """Filters a JSON list based on a provided list of IDs.

  Args:
      data: A list of dictionaries representing the JSON data.
      list_ids: A list of IDs to filter by.

  Returns:
      A new list containing only the items with matching IDs.
  """
  filtered_data = [item for item in data if item["id"] in list_ids]
  return filtered_data



list_ids = ['000000053864', '000000422064', '000000090341', '000000071126', '000000011328', '000000025649', '000000279632', '000000428508', '000000448930', '000000461989', '000000506463', '000000359420', '000000202112', '000000026829', '000000562428', '000000213534',
'000000569251', '000000493526', '000000169646', '000000369770', '000000213146', '000000414747', '000000343610', '000000110735', '000000268406', '000000465468', '000000358667', '000000455350', '000000021260', '000000434765', 'VG_100K_2/1847', 'VG_100K/498135', 
'VG_100K_2/2407359', 'VG_100K_2/2401086', 'VG_100K_2/2389714', 'VG_100K_2/2389235', 'VG_100K_2/2378917', 'VG_100K/2377029', 'VG_100K/2375290', 'VG_100K/2367053', 'VG_100K/2363742', 'VG_100K/2341597', 'VG_100K/2332218', 'VG_100K/2315869', 2321654, 2409688, 2339761, 2318778,
2383111, 2404346, 2372508, '331912174X', '1942878117', '1907621016', '1572301317', '936070463', '128016086', '1893910245', '393337146', '520040228', '1566590523', '1856175472', '030018462X', '857687786',
'000000241897', '000000496128', '000000562174', '000000205520', '000000140865', '000000174794', '000000404502', '000000184673', '000000067779', '000000046563', '000000412002', '000000157371', '000000034469', '000000174758', '000000193954', '000000327241',
'000000337808', '000000471690', '000000487870', '000000372829', '000000127006', '000000374800', '000000179926', '000000486620', '000000562575', '000000551172', '000000404243', '000000435260', '000000156986', '000000544893', '000000127202', '000000556643',
'lRG0ZjQ_0', 'bdnBMrH_0', '1TtiVYw_0', 'fK84xXb_0', 'A5kislv_0', 'EieQ5BH_0', '50nV0Lu_0', 'kU0Ma9E_0', 'f7jfeo9_0', 'rpD8Fnb_0', 'JLgAoSE_0', 'qLOWrKn_0', 'IdWHnf6_0', 'iKdsZ3E_0', 'vw9lXdp_0', 'wdY7TL2_0',
'9923fac0a926923e', 'c4dee4e9d981fa1c', 'RhkvFaO_0', 'GjUumVV_0', 'HSJi4eb_0', 'y7a6QQX_0', 'XOsTkDm_0', 'pJl35Wg_0', 'QkMBtuK_0', 'Tfhjd8j_0', '15xDvgw_0', 'MD6NWLX_0', 'e0lW2wr_0', 'HtTgipR_0', 'GYKw6oF_0', 'YMRyrwp_0',
]
data_path = '/data/jieneng/data/llava_datasets/LLaVA-Tuning/llava_v1_5_mix665k.json'
data = json.load(open(data_path, "r"))
filtered_data = filter_json_by_id(data, list_ids)
# Write the filtered data to a new JSON file
with open('filtered_llava_v1_5_mix665k.json', 'w') as outfile:
  json.dump(filtered_data, outfile, indent=4)

print("Filtered data written to filtered_data.json")