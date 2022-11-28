from typing import Union
from dotenv import load_dotenv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

import os

# Environment variables.
load_dotenv()

# Authenticate to Service account Google Drive.
gauth = GoogleAuth()

gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
    os.path.join(os.path.dirname(__file__), os.environ['JSON']),
    scopes=[os.environ['SCOPES']]
)

# Instance the Google Drive main class.
drive = GoogleDrive(gauth)


def retrieve_data_from_Drive(
    folder_name: Union[str, None]
) -> None:
    """
    Extract all the files from a given folder.
    """
    # Get the list of all folders in the database.
    folder_list = drive.ListFile(
        {'q': "'root' in parents and trashed=false"}
        ).GetList()

    # Folder id.
    folder_id = None

    for folder in folder_list:
        if folder['title'] == folder_name:
            folder_id = folder['id']
            print('title: %s, id: %s' % (folder['title'], folder['id']))
            print("--------------------------------------------------------")

    if folder_id:
        file_list = drive.ListFile(
            {'q': "'{}' in parents and trashed=false".format(folder_id)}
        ).GetList()

    file_iterador = sorted(file_list, key=lambda x: x['title'])
    for i, file in enumerate(file_iterador, start=1):
        print('Downloading {} from GDrive ({}/{})'.format(
            file['title'], i, len(file_list))
        )
        file.GetContentFile(file['title'])


def see_data_from_Drive() -> None:
    """
    Visualize all the folders in the database.
    """
    # Get the list of all folders in the database.
    folder_list = drive.ListFile(
        {'q': "'root' in parents and trashed=false", 'orderBy': 'title'}
    ).GetList()

    for folder in folder_list:
        print('title: %s, id: %s' % (folder['title'], folder['id']))
        print("----------------------------------------------------")


def delete_folder_from_Drive(
    folder_id: Union[str, None]
) -> None:
    """
    Delete the folder with the name folder_name.
    """
    folders = drive.ListFile(
        {'q': "'root' in parents and trashed=false"}
    ).GetList()

    counter = 0
    for folder in folders:
        if (folder['id'] == folder_id):
            counter += 1
            folder.Delete()
            print("Folder: %s with id: %s... successfully deleted." % (
                    folder['title'], folder['id'][0:int(len(folder['id'])/2)]
                )
            )
    if counter == 0:
        print("Folder not found on Drive.")


def rename_folder_from_Drive(
    folder_id: str,
    new_name: str
) -> None:
    """
    Renames the folder with folder_id by the name new_name.
    """
    try:
        folder = drive.auth.service.files().get(fileId=folder_id).execute()
        folder["title"] = new_name
        drive.auth.service.files().update(
            fileId=folder_id, body=folder
        ).execute()
    except AttributeError:
        print("Folder not found in Drive. \n")

see_data_from_Drive()