import os
import pandas as pd
import argparse
import numpy as np

class post_processing:

    def process_single_csv(self, csv_file):

        print("Processing CSV file: {}".format(csv_file))
        # Read the CSV file
        df = pd.read_csv(csv_file)

        print("DF head: ", df.head())

        print("DF columns: ", df.columns)

        # Check if 'FileName' column exists
        if 'FileName' not in df.columns:
            raise ValueError("The CSV file does not contain a 'FileName' column.")

        # Create a new column 'Folder' by splitting the 'FileName' column on the underscore and taking the first part
        df['Folder'] = df['FileName'].apply(lambda x: x.rsplit('_', 1)[0])

        print("Folder column created")

        print("DF columns: ", df.columns)

        # Group the DataFrame by the 'Folder' column
        grouped = df.groupby('Folder')
        processed_df = pd.DataFrame()

        for name, group in grouped:
            group = group.reset_index(drop=True)
            group = self.post_process(group)
            processed_df = pd.concat([processed_df, group], axis = 0).reset_index(drop=True)

        print("Processed DataFrame head: ", processed_df.head())
        print("Processed DataFrame columns: ", processed_df.columns)

        processed_df = processed_df.drop(columns=['Folder'])
        processed_df.to_csv("processed.csv", index=False)

    def process_csv_files(self, root_dir):
        # Iterate over each folder in the root directory
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            # Check if the folder contains a CSV file
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            for csv_file in csv_files:
                # Read the CSV file
                df = pd.read_csv(os.path.join(folder_path, csv_file))

                # Post process the DataFrame
                df = self.post_process(df)

                # Save the updated CSV file
                df.to_csv(os.path.join(folder_path, csv_file), index=False)

    
    def propogate_anesthetic(self, data):
        # If 'anesthetic' label exists in 'Overall Task' column, make the above 4 rows and under 4 rows 'Overall Task' column all 'anesthetic'
        changes_made = 0
        anesthetic_indices = data[data['Overall Task'] == 'anesthetic'].index
        for index in anesthetic_indices:
            if index >= 4:
                data.loc[index - 3:index, 'Overall Task'] = 'anesthetic'
            if index <= len(data) - 4:
                data.loc[index:index + 3, 'Overall Task'] = 'anesthetic'
        return data
    
    def fix_insert_catherer(self, data):
        # if dialator is within 10 frames of insert catheter, change it to insert catheter

        # find the indices of 'insert_catheter' and 'dilator'
        insert_catheter_indices = data[data['Overall Task'] == 'insert_catheter'].index
        dilator_indices = data[data['Overall Task'] == 'dilator'].index
        # iterate over the indices of 'insert_catheter'
        for index in insert_catheter_indices:
            # check if there is a 'dilator' within 10 frames of the 'insert_catheter'
            if index - 20 in dilator_indices:
                # change the 'dilator' to 'insert_catheter' and all previous connecting 'dialator' frames
                data.loc[index - 20:index, 'Overall Task'] = 'insert_catheter'
                # check for previous connnecting dialator indicies
                cur_index = index - 20
                while cur_index > 0:
                    if data.loc[cur_index - 1, 'Overall Task'] == 'dilator':
                        data.loc[cur_index - 1, 'Overall Task'] = 'insert_catheter'
                        cur_index -= 1
                    elif data.loc[cur_index - 10, 'Overall Task'] == 'dilator':
                        data.loc[cur_index - 10:cur_index, 'Overall Task'] = 'insert_catheter'
                        cur_index -= 10
                    else:
                        break
                   

        return data

    def replace_second_instance_dialator(self, data):
        # Find dialator indices
        dilator_indices = data[data['Overall Task'] == 'dilator'].index

        for index in dilator_indices:
            # find indices spaced by > 10 frames
            if index - 10 in dilator_indices:
                # change the 'dilator' to 'insert_catheter' and all previous connecting 'dialator' frames
                data.loc[index - 10:index, 'Overall Task'] = 'insert_catheter'
                # check for previous connnecting dialator indicies
                cur_index = index - 10
                while cur_index > 0:
                    if data.loc[cur_index - 1, 'Overall Task'] == 'dilator':
                        data.loc[cur_index - 1, 'Overall Task'] = 'insert_catheter'
                        cur_index -= 1
                    elif data.loc[cur_index - 10, 'Overall Task'] == 'dilator':
                        data.loc[cur_index - 10:cur_index, 'Overall Task'] = 'insert_catheter'
                        cur_index -= 10
                    else:
                        break

        return data
    
    def remove_3_frame_labels(self, data):
        # Remove labels that are present in only 3 frames
        # data = data.reset_index(drop=True)
        for index, row in data.iterrows():
            # check if the one before and after the current index is the same
            if index > 2 and index < len(data) - 2:
                if data.loc[index - 2, 'Overall Task'] == data.loc[index + 2, 'Overall Task']:
                    data.loc[index, 'Overall Task'] = data.loc[index - 2, 'Overall Task']
                # else:
                #     data.loc[index, 'Overall Task'] = data.loc[index, 'Overall Task']
        return data
    
    def replace_long_axis_with_cross_section(self, data):
        # if long axis is in the first 100 frames, change it to cross-section
        if data.loc[:100, 'Overall Task'].str.contains('Long-axis').any():
            data.loc[:100, 'Overall Task'] = data.loc[:100, 'Overall Task'].replace('Long-axis', 'Cross-section')

        return data
    
    def fill_nothing_between_insert_catheter(self, data):
        # Find the indices of 'insert_catheter' in the 'Overall Task' column
        insert_catheter_indices = data[data['Overall Task'] == 'insert_catheter'].index

        # Iterate over the indices of 'insert_catheter'
        for index in insert_catheter_indices:
            # Check if previous catheter is within 10 frames
            if index - 35 in insert_catheter_indices:
                # change in between frames to insert_catheter
                data.loc[index - 35: index, 'Overall Task'] = 'insert_catheter'

        return data

    def post_process(self, data):
        data = self.propogate_anesthetic(data)
        data = self.remove_3_frame_labels(data)
        data = self.replace_second_instance_dialator(data)
        data = self.fill_nothing_between_insert_catheter(data)
        data = self.replace_long_axis_with_cross_section(data)
        return data
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", type=str, help="Specify the root directory path")
    parser.add_argument("-csv", "--main_csv_file", type=str, help="Specify the main csv path")
    args = parser.parse_args()

    pp = post_processing()

    if args.root_dir:
        pp.process_csv_files(args.root_dir)
    elif args.main_csv_file:
        pp.process_single_csv(args.main_csv_file)