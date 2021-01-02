# Package_Items
#   Miguel Tavera
#   2020-12-31
#   Basd on Eric Donders code - Read items information and construct sets for use in Auto Dynamax Adventures code


from copy import copy
import csv, pickle

item_list = []
with open('Pokemon_Data/items.txt', newline='\n') as file:
    spamreader = csv.reader(file)
    for row in spamreader:
        item_list.append(row)
print('Read and processed items file.')


with open('Pokemon_Data/Items.pickle', 'wb') as file:
    pickle.dump(item_list, file)
print('Finished packaging Items!')

