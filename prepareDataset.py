import pandas
import os
import argparse
import shutil

FLAGS = None

def unpackZipFiles(dataLocation,datasetType):

    if datasetType == "Train":
        for i in range(1, 11):
            print("Extracting data from: Training_Data_Part{}.zip".format(i))
            extractionDir = os.path.join(dataLocation, "Training_Data_Part{}".format(i))
            if not os.path.exists(extractionDir):
                os.mkdir(extractionDir)
            zipFile = os.path.join(dataLocation, "Training_Data_Part{}.zip".format(i))
            shutil.unpack_archive(zipFile,extractionDir)

    elif datasetType == "Test":
        print("Extracting data from: Test_Data.zip")
        extractionDir = os.path.join(dataLocation, "Test_Data")
        if not os.path.exists(extractionDir):
            os.mkdir(extractionDir)
        zipFile = os.path.join(dataLocation, "Test_Data.zip")
        shutil.unpack_archive(zipFile, extractionDir)

    elif datasetType == "Unlabelled":
        for i in range(1,9):
            print("Extracting data from: Unlabelled_Data_Part{}.zip".format(i))
            extractionDir = os.path.join(dataLocation, "Unlabelled_Data_Part{}".format(i))
            if not os.path.exists(extractionDir):
                os.mkdir(extractionDir)
            zipFile = os.path.join(dataLocation, "Unlabelled_Data_Part{}.zip".format(i))
            shutil.unpack_archive(zipFile, extractionDir)

def moveFilesToMainDirectory(videoDir,mainDirectory):
    videoID = os.path.basename(videoDir)
    print("Transferring data from video: {}".format(videoID))
    newlocation = os.path.join(mainDirectory,videoID)
    if not os.path.exists(newlocation):
        os.mkdir(newlocation)
    fileNames = os.listdir(videoDir)
    for file in fileNames:
        oldFileLocation = os.path.join(videoDir,file)
        newFileLocation = os.path.join(newlocation,file)
        shutil.move(oldFileLocation,newFileLocation)

def createMainDatasetCSV(mainDirectory,datasetType):
    print("Creating main dataset csv")

    if datasetType == "Train":
        dataCSVFileName = "Training_Data.csv"
    elif datasetType == "Test":
        dataCSVFileName = "Test_Data.csv"
    else:
        dataCSVFileName = "Unlabelled_Data.csv"

    videoIDs = [x for x in os.listdir(mainDirectory) if not "." in x]
    i = 0
    dataCSVInitialized = False
    for video in videoIDs:
        labelFile = pandas.read_csv(os.path.join(mainDirectory,video,video+"_Labels.csv"))        
        labelFile["Folder"] = [os.path.join(mainDirectory,video) for i in range(len(labelFile.index))]
                
        if not dataCSVInitialized:
            dataCSV = pandas.DataFrame(columns=labelFile.columns)
            dataCSVInitialized = True
        dataCSV = pandas.concat([dataCSV,labelFile])
    for column in dataCSV.columns:
        if "Unnamed" in column:
            dataCSV = dataCSV.drop(column,axis=1)
    dataCSV.to_csv(os.path.join(mainDirectory,dataCSVFileName),index = False)

def checkAllPresent(dataLocation,datasetType):
    missingFiles = []
    if datasetType == "Train":
        for i in range(1, 11):
            zipFile = os.path.join(dataLocation, "Training_Data_Part{}.zip".format(i))
            if not os.path.exists(zipFile):
                missingFiles.append("Training_Data_Part{}.zip".format(i))
    elif datasetType == "Test":
        zipFile = os.path.join(dataLocation, "Test_Data.zip")
        if not os.path.exists(zipFile):
            missingFiles.append("Test_Data.zip")
    elif datasetType == "Unlabelled":
        for i in range(1,9):
            zipFile = os.path.join(dataLocation, "Unlabelled_Data_Part{}.zip".format(i))
            if not os.path.exists(zipFile):
                missingFiles.append("Unlabelled_Data_Part{}.zip".format(i))
    if len(missingFiles)>0:
        print("Missing the following files:")
        for file in missingFiles:
            print("\t{}".format(file))
        exit()

def splitTrainingData(mainDirectory):

    print("Applying dataset split...")

    for df_name in ['train', 'val', 'test']:
        dir_df = pandas.read_csv(os.path.join('split_csvs', f'{df_name}.csv'), index_col=False)
        dir_path = os.path.join(mainDirectory, df_name)

        os.mkdir(dir_path)

        for index, row in dir_df.iterrows():
            vname = row['VideoName']
            print(f"Transferring data from video: {vname} (frames {row['first_file']} to {row['last_file']})")

            from_dir = os.path.join(mainDirectory, vname)
            to_dir = os.path.join(dir_path, vname)

            if os.path.exists(to_dir):
                to_dir = to_dir + "_1"   # ex. AN01-20210104-154854 -> AN01-20210104-154854_1

            os.mkdir(to_dir)

            for fname in os.listdir(from_dir):
                if not fname.endswith('.csv') and row['first_file'] <= fname and fname <= row['last_file']:
                    from_path = os.path.join(from_dir, fname)
                    to_path = os.path.join(to_dir, fname)
                    shutil.move(from_path, to_path)

            # Handle csv
            csv_file = [fname for fname in os.listdir(from_dir) if fname.endswith('.csv')][0]
            orig_df = pandas.read_csv(os.path.join(from_dir, csv_file))

            print("Copying CSV...")
            start_split_idx = orig_df.index[orig_df['FileName'] == row['first_file']][0]
            end_split_idx = orig_df.index[orig_df['FileName'] == row['last_file']][0]
            new_df = orig_df.iloc[start_split_idx:end_split_idx + 1]
            new_df.to_csv(os.path.join(to_dir, csv_file))
    
        createMainDatasetCSV(dir_path, 'Train')

    for dir_name in os.listdir(mainDirectory):
        if dir_name not in ['train', 'val', 'test']:
            shutil.rmtree(os.path.join(mainDirectory, dir_name))
        
            

def createDataset():
    baseLocation = FLAGS.compressed_location
    targetLocation = FLAGS.target_location
    datasetType = FLAGS.dataset_type
    checkAllPresent(baseLocation,datasetType)
    unpackZipFiles(baseLocation,datasetType)

    if datasetType == "Train":
        dataSetLocation = os.path.join(targetLocation,"Training_Data")
        if not os.path.exists(dataSetLocation):
            os.mkdir(dataSetLocation)
        for i in range(1, 11):
            dataFolder = os.path.join(baseLocation, "Training_Data_Part{}".format(i))
            for videoDir in os.listdir(dataFolder):
                moveFilesToMainDirectory(os.path.join(dataFolder, videoDir), dataSetLocation)
            shutil.rmtree(dataFolder)
            print("Removed empty directory {}".format(dataFolder))


    elif datasetType == "Test":
        dataSetLocation = os.path.join(targetLocation,"Test_Data")
        if not os.path.exists(dataSetLocation):
            os.mkdir(dataSetLocation)
        dataFolder = os.path.join(baseLocation, "Test_Data")
        for videoDir in os.listdir(dataFolder):
            moveFilesToMainDirectory(os.path.join(dataFolder, videoDir), dataSetLocation)
        shutil.rmtree(dataFolder)
        print("Removed empty directory {}".format(dataFolder))

    elif datasetType == "Unlabelled":
        dataSetLocation = os.path.join(targetLocation,"Unlabelled_Data")
        if not os.path.exists(dataSetLocation):
            os.mkdir(dataSetLocation)
        for i in range(1, 9):
            dataFolder = os.path.join(baseLocation, "Unlabelled_Data_Part{}".format(i))
            for videoDir in os.listdir(dataFolder):
                moveFilesToMainDirectory(os.path.join(dataFolder, videoDir), dataSetLocation)
            shutil.rmtree(dataFolder)
            print("Removed empty directory {}".format(dataFolder))

    else:
        print("Unrecognized dataset type. Must be one of: Train, Test or Unlabelled")
        

    ########################################### ADDED ############################################

    if datasetType == 'Train' and FLAGS.apply_split:
        splitTrainingData(dataSetLocation)
    else:
        createMainDatasetCSV(dataSetLocation,datasetType)

    ################################################################################################



    print("Dataset preparation complete. Data located in directory: {}".format(dataSetLocation))



if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--compressed_location',
      type=str,
      default='',
      help='Name of the directory where the compressed data files are located'
  )
  parser.add_argument(
      '--target_location',
      type=str,
      default='',
      help='Name of the directory where the uncompressed data files will be located'
  )
  parser.add_argument(
      '--dataset_type',
      type=str,
      default='Train',
      help='Type of Dataset you are creating: should be Train, Test, or Unlabelled'
  )

  parser.add_argument(
      '--apply_split',
      action="store_true",
      help='If provided, split data into three folders - train, val, and test'
  )

  FLAGS, unparsed = parser.parse_known_args()
  createDataset()