import csv
import os
import gc

import pandas as pd

class logger():
    def __init__(self):
        self.folder = f"Data/"
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.folder, exist_ok=True)
    def write(self, message, file_no = 0):
        filenames = os.path.join(self.folder, f"Raw_{file_no}.txt")
        with open(filenames, 'a', newline='') as f:
            writer = csv.writer(f)

            if isinstance(message, (list, tuple)):
                writer.writerow(message)
            else:
                writer.writerow([message])
        return

# log = logger()

class DataReader(object):
    def __init__(self, splits = 5):
        self.labels_df = pd.read_csv("Data/train_labels.csv")
        # print(labels_df.shape)
        # print(labels_df.head())
        no_of_customers = self.labels_df.shape[0]
        size = no_of_customers // splits
        sizes = [size for i in range(splits-1)] + [size + no_of_customers % size]
        # print(sizes)

        self.index = [sum(sizes[:i]) for i in range(splits+1)]
        # print(index)

        self.cus_ids_per_set = []
        for i in range(splits):
            df = self.labels_df.iloc[self.index[i]:self.index[i+1]]

            self.cus_ids_per_set.append((df.iloc[0].values[0],df.iloc[-1].values[0]))

            label_dist = df["target"].value_counts()
            print(f" 0: {label_dist[0]}, 1: {label_dist[1]}, ratio: {label_dist[0]/label_dist[1]}")

        # for pair in self.cus_ids_per_set:
        #     print(pair)

    def get_dataset(self, set_id = 0):
        if set_id >= 5:
            print("Out of range")
            return 0, 0
        file_path = "Data/train_data.csv"
        start_cus_id = self.cus_ids_per_set[set_id][0]
        end_cus_id = self.cus_ids_per_set[set_id][1]

        data_list = []
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)

            read = False
            last = False

            # Gets the header from the first row
            header = next(iter(csv_reader))

            for line in csv_reader:
                if line[0] == start_cus_id and not read:
                    read = True
                    print("Found the starting customer id")

                if line[0] == end_cus_id and not last:
                    last = True
                    print("Reached the end customer id")

                if read:
                    if last and line[0] != end_cus_id:
                        break
                    data_list.append(line)


        train_df = pd.DataFrame(data_list, columns=header)
        labels_df = self.labels_df.iloc[self.index[set_id]:self.index[set_id+1]]
        return train_df, labels_df

split_data_into = 5

DR = DataReader(split_data_into)

os.makedirs("SplitData", exist_ok=True)

for i in range(split_data_into):
    print(f"Split: {i}")
    train_df, labels_df = DR.get_dataset(i)

    merged_df = pd.merge(train_df, labels_df, on="customer_ID")
    merged_df.to_csv(f"SplitData/train_data_{i}.csv", index=False)

    del train_df, labels_df, merged_df
    gc.collect()