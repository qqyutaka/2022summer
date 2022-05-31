# Copyright 2022 AudioT.  All rights reserved.
# This code is proprietary to AudioT, INC and provided to OMSA students under the conditions of the
# AudioT NDA terms. 

import os
import sys

import boto3
import pandas
import datetime
import boto3
import concurrent.futures
import pandas as pd
import json
'''
This S3 Utility class provides functions that allow for listing and download of files in AWS S3. Packages needed
include Boto3 and pandas. 

IMPORTANT: Edit the config file in the directory /config to add the necessary credentials.

Credits: M. Rizwan for providing his own utility functions for AudioT use. 
'''


class S3_Utility:
    def __init__(self, aws_access_key, aws_secret_key, region='us-east-1'):
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.s3_resource = boto3.resource('s3',
                                          aws_access_key_id=aws_access_key,
                                          aws_secret_access_key=aws_secret_key,
                                          region_name=region
                                          )
        self.bucket_list = [each.name for each in self.s3_resource.buckets.all()]

    def get_buckets(self):
        return self.s3_resource.buckets.all()

    def s3_list_files(self, bucket_name=None, Directory='', FileExtension='', Path=True):
        """
        This method returns a list of files with a specified file extension.
        Args:
            bucket_name: Bucket which files are to be listed
            Directory: the path/folder to specify which folder to list
            FileExtension: types of files specified to be listed (only one is valid)
            Path: Defines whether to return path of file (defaulted true)

        Returns:

        """

        if (bucket_name is not None) and (bucket_name in self.bucket_list):
            filename = []
            if FileExtension != '':
                if not (Path):
                    filenames = [each.key.split('/')[-1] for each in
                                 self.s3_resource.Bucket(bucket_name).objects.filter(Prefix=Directory).all() if
                                 each.key.split(".")[-1] == FileExtension]
                else:
                    filenames = [each.key for each in
                                 self.s3_resource.Bucket(bucket_name).objects.filter(Prefix=Directory).all() if
                                 each.key.split(".")[-1] == FileExtension]

            else:
                if not (Path):
                    filenames = [each.key.split('/')[-1] for each in
                                 self.s3_resource.Bucket(bucket_name).objects.filter(Prefix=Directory).all()]
                else:
                    filenames = [each.key for each in
                                 self.s3_resource.Bucket(bucket_name).objects.filter(Prefix=Directory).all()]

            return filenames
        else:
            print("Bucket Name is not correct!")
            return []

    def s3_download_files(self, bucket_name=None, remote_path_list=[], localpath=None, isWriteSummary=True):
        """
        This function downloads a file from a specified bucket.
        Args:
        bucket_name: Bucket to be downloaded from.
        remote_path_list: location of file in the bucket.
        localpath: is path in current working directory of where you want to store files locally
        isWriteSummary:

        Returns:

        """

        if localpath is None:
            localpath = ''
        if type(remote_path_list) == list and remote_path_list != []:
            if (bucket_name is not None) and (bucket_name in self.bucket_list):
                if not os.path.exists(os.path.join(os.getcwd(), localpath)):
                    os.makedirs(localpath)
                out = []
                CONNECTIONS = 200  # setting for how many concurrent downloads you want, currently it is getting 200 downloads concurrently?
                TIMEOUT = 60
                bucket_obj = self.s3_resource.Bucket(bucket_name)
                config = [(bucket_obj, each, os.path.join(os.getcwd(), localpath, each.split('/')[-1])) for each in
                          remote_path_list]

                def load_url(bkt, remote_file_path, local_path, timeout):
                    check = [each.key for each in bkt.objects.filter(Prefix=remote_file_path).all()]
                    if check != []:
                        bkt.download_file(Key=remote_file_path, Filename=local_path)
                        return [bkt.name, remote_file_path, 1]
                    else:
                        print("Bucket: " + bkt.name + " does not have this file: " + remote_file_path + " !")
                        return [bkt.name, remote_file_path, 0]

                with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
                    future_to_url = (executor.submit(load_url, each[0], each[1], each[2], TIMEOUT) for each in config)
                    # time1 = time.time()
                    for future in concurrent.futures.as_completed(future_to_url):
                        try:
                            data = future.result()
                        except Exception as exc:
                            data = str(type(exc))
                        finally:
                            out.append(data)
                            # print(str(len(out)),end="\r")

                        # time2 = time.time()

                # print(f'Took {time2-time1:.2f} s')

                if isWriteSummary:
                    print(out)
                    pd.DataFrame(out, columns=['BucketName', 'BucketFilePath', 'Status']).to_csv(
                        os.path.join(localpath, "Download_Summary.csv"), index=False)
            else:
                print("Bucket Name is not correct!")
        else:
            print("Remote Path arguement must be a list and cannot be empty!")


def main():
    path_to_cred_file = os.path.join("config", 'AWS_S3_creds.json')
    aws_creds = 0
    with open(path_to_cred_file, 'r') as f:
        aws_creds = json.loads(f.read())
    obj = S3_Utility(aws_access_key=aws_creds['AWS_ACCESS_KEY_ID'], aws_secret_key=aws_creds['AWS_SECRET_ACCESS_KEY'],
                     region='us-east-1')

    bucket_name = 'audiot-disk03'
    # 1 - LIST OF BUCKETS
    print(len(obj.bucket_list))
    print(obj.bucket_list)

    # 2 - LIST OF FILES IN A GIVEN DIRECTORY
    folder_to_list = 'TRF0_2020-12/trf0-recorder-3/2020-12-17/00/'
    list_of_files = obj.s3_list_files(bucket_name, Directory=folder_to_list, FileExtension='', Path=True)
    print(list_of_files)

    # 3 - DOWNLOAD FILES FROM A FILE LIST ABOVE
    obj.s3_download_files(bucket_name=bucket_name, remote_path_list=list_of_files, localpath=None, isWriteSummary=True)
    # !!!! ^^^^ Heads up, this will download a lot files....

    sys.exit()


if __name__ == '__main__':
    main()
